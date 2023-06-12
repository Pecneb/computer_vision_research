"""
    Predicting trajectories of objects
    Copyright (C) 2022  Bence Peter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact email: ecneb2000@gmail.com
"""
# disable sklearn warning
def warn(*arg, **args):
    pass
from joblib import dump
import warnings
warnings.warn = warn
import cv2 as cv
import argparse
import time
from pathlib import Path
from dataManagementClasses import Detection
from deepsortTracking import (
    initTrackerMetric, 
    getTracker, 
    updateHistory
)
import databaseLogger as databaseLogger
import os
import tqdm
from masker import masker
from processing_utils import downscale_TrackedObjects

def parseArgs():
    """Function for Parsing Arguments

    Returns:
        args.input: input video source given in command line argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='+', help="Path to video.")
    parser.add_argument("--k_velocity", type=int, default=10, help="K value for differentiation of velocity.")
    parser.add_argument("--k_acceleration", type=int, default=2, help="K value for differentiation of acceleration.")
    parser.add_argument("--output", help="Path of output directory.", required=True)
    parser.add_argument("--history", default=0, type=int, help="Length of history for regression input. WARNING: TO SAVE JOBLIB DATABASES PROPERLY THIS HAS TO BE SET REALLY HIGH (EG. 1000<HISTORY)")
    parser.add_argument("--future", default=0, type=int, help="Length of predicted coordinate vector.")
    parser.add_argument("--k_trainingpoints", default=0, type=int, help="The number how many coordinates from the training set should be choosen to train with.")
    parser.add_argument("--degree", default=0, type=int, help="Degree of polynomial features used for Polynom fitting.")
    parser.add_argument("--max_cosine_distance", type=float, default=10.0,
                        help="Gating threshold for cosine distance metric (object appearance).")
    parser.add_argument("--max_iou_distance", default=0.7, type=float)
    parser.add_argument("--nn_budget", type=float, default=100,
                        help="Maximum size of the appearance descriptors gallery. If None, no budget is enforced.")
    #parser.add_argument("--min_detection_height", type=float, default=0,
    #                    help="Threshold on the detection bounding box height. Detections with height smaller than this value are disregarded")
    #parser.add_argument("--min_confidence", type=float, default=0.7,
    #                    help="Detection confidence threshold. Disregard all detections that have a confidence lower than this value.")
    #parser.add_argument("--nms_max_overlap", type=float, default=1.0,
    #                    help="Non-maxima suppression threshold: Maximum detection overlap.")
    parser.add_argument("-d", "--device", default='cuda', help="Choose device to run neuralnet. example: cpu, cuda, 0,1,2,3...")
    parser.add_argument("--yolov7", default=1, type=int, help="Choose which yolo model to use. Choices: yolov7 = 1, yolov4 = 0")
    parser.add_argument("--resume", "-r", action="store_true", help="Use this flag if want to resume video from last session left off.")
    parser.add_argument("--show", action="store_true", default=False, help="Use this flag to display video while running detection, prediction on video.")
    args = parser.parse_args()
    return args

def printDetections(detections):
    """Function for printing out detected objects

    Args:
        detections (tuple): _description_
    """
    for obj in detections:
        # bbox: x, y, w, h
        print("Class: {}, Confidence: {}, Position: {}".format(obj[0], obj[1], obj[2]))

def getTargets(detections, frameNum, targetNames=['car, person']):
    """Function to extract detected objects from the detections, that labels are given in the targetNames argument

    Args:
        detections ((str, float, (int, int, int, int))): Return value of the darknet neuralnet
        targetNames (list, optional): List of labels, that should be tracked. Defaults to ['car', 'person'].

    Returns:
        targets (list[Detection]): Returns a list of Detection objects
    """
    targets = []
    for label, conf, bbox in detections:
        # bbox: x, y, w, h
        if label in targetNames:
            targets.append(Detection(label, conf, bbox[0], bbox[1], bbox[2], bbox[3], frameNum))
    return targets

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def draw_boxes(history, image, colors, frameNumber):
    """Draw detection information to video output

    Args:
        history (list[TrackedObject]): Tracking history
        image (OpencvImage): input image, to draw detection information, on
        colors (dict[key,value]): dictionary containing color and class_name pairs
        frameNumber (int): current frame's number

    Returns:
        OpencvImage: image with the detection information on it
    """
    for detections in history:
        detection = detections.history[-1]
        if detection.frameID == frameNumber:
            bbox = (detection.X, detection.Y, detection.Width, detection.Height)
            left, top, right, bottom = bbox2points(bbox)
            cv.rectangle(image, (left, top), (right, bottom), colors[detection.label], 1)
            cv.putText(image, "{} ID{} [{:.2f}] VX: {:.2f} VY: {:.2f}".format(detection.label, detections.objID, float(detection.confidence), float(detection.VX), float(detection.VY)),
                        (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[detection.label], 2)

def num_of_moving_objs(history: list) -> int:
    n = 0
    for obj in history:
        if obj.isMoving:
            n += 1
    return n

def log_to_stdout(*args):
    print('\n' * 200) 
    for arg in args:
        if type(arg) is list:
            for ar in arg:
                print(ar)
        else:
            print(arg)
    print('\n'*5)

def generateOutputName(input, outdir):
    """Generate output path from input path and output directory.

    Args:
        input (str): Input path. 
        outdir (str): Output directory path. 

    Returns:
        Path: Output file path. 
    """
    inputPath = Path(input)
    outdirPath = Path(outdir)
    outputPath = outdirPath.joinpath(inputPath.stem)
    return outputPath

def getDirectoryEntries(dirpath):
    path = Path(dirpath)
    inputs = []
    for p in path.glob("*.mp4"):
        inputs.append(str(p))
        print(p)
    return inputs

def main():
    args = parseArgs()

    if Path(args.input[0]).is_dir():
        inputs = getDirectoryEntries(args.input[0])
    else:
        inputs = args.input
        print(args.input)
    
    outdir = Path(args.output)
    if not outdir.exists():
        outdir.mkdir()

    if args.yolov7:
        import yolov7api as yolov7api 
    else:
        import hldnapi as hldnapi
        from darknet import class_colors

    # buffer to log at the end
    buffer2log = []

    # generating colors for bounding boxes based on the class names of the neural net
    if args.yolov7:
        colors = yolov7api.COLORS 
    else:
        colors = hldnapi.colors

    # log metadata to database
    if args.yolov7:
        yoloVersion = '7'
    else:
        yoloVersion = '4'

    cap = cv.VideoCapture(inputs[0]) # retrieve an initial image from video to create mask

    # create mask, so only in the area of interest will be used in detection 
    _, img = cap.read()
    mask = masker(img)

    for input in tqdm.tqdm(inputs, desc="Videos"):
        outputName = generateOutputName(input, args.output) # generate output name

        path2db = outputName.parent / (outputName.name + ".db") # add .db suffix to output name fo SQL db
        path2db = databaseLogger.init_db(path2db) # initialize database
        print(f"SQL DB path: {path2db}")

        print(path2db)
        # create database connection
        db_connection = databaseLogger.getConnection(path2db)

        # generate joblib file output path
        path2joblib = outputName.parent / (outputName.name + ".joblib")
        print(f"Joblib DB path: {path2joblib}")

        # If joblib database is already exists and video is requested to be resumed, 
        # load existing data and continue detection where it was left off
        if os.path.exists(path2joblib) and args.resume:
            from processing_utils import load_joblib_tracks
            buffer2joblibTracks = load_joblib_tracks(path2joblib) 
        buffer2joblibTracks = []

        # device is still hardcoded, so a gpu with cuda capability is needed for now, for real time speed it is necessary
        databaseLogger.logMetaData(db_connection, args.history, args.future, yoloVersion, "gpu", yolov7api.IMGSZ, yolov7api.STRIDE, yolov7api.CONF_THRES, yolov7api.IOU_THRES)
        databaseLogger.logRegression(db_connection, "LinearRegression", "Ridge", args.degree, args.k_trainingpoints)

        # get video capture object
        cap = cv.VideoCapture(input)

        # get frame width
        frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        # get frame height
        frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        # exit if video cant be opened
        if not cap.isOpened():
            print("Source cannot be opened.")
            exit(0)

        # forward declaration of history(list[TrackedObject])
        trackedObjects = []

        # resume video where it was left off, if resume flag is set
        if args.resume:
            lastframeNum = databaseLogger.getLatestFrame(db_connection)
            if lastframeNum > 0 and lastframeNum < cap.get(cv.CAP_PROP_FRAME_COUNT):
                cap.set(cv.CAP_PROP_POS_FRAMES, lastframeNum-1) 
            # create DeepSortTracker with command line arguments, pass db_connection to query last objID from database
            tracker = getTracker(initTrackerMetric(args.max_cosine_distance, args.nn_budget), historyDepth=args.history, db_connection=db_connection, max_iou_distance=args.max_iou_distance)
        else:
            lastframeNum = 0
            # DeepSortTracker without db_connection, starts objID count from 1
            tracker = getTracker(initTrackerMetric(args.max_cosine_distance, args.nn_budget), historyDepth=args.history, max_iou_distance=args.max_iou_distance)

        print(f"Starting video from frame number: {lastframeNum}")
        # start main loop
        for frameIDX in tqdm.tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))), initial=lastframeNum):
            try:
                # things to log to stdout
                to_log = []

                # get current frame from video
                ret, frame = cap.read()
                if frame is None:
                    print("Video ended, closing player.")
                    break

                # get current frame number
                frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)

                # time before computation
                prev_time = time.time()

                # run yolo inference 
                if args.yolov7:
                    detections = yolov7api.detect(cv.bitwise_or(frame, frame, mask=mask)) 
                else:
                    detections = hldnapi.cvimg2detections(cv.bitwise_or(frame, frame, mask=mask))

                # filter detections, only return the ones given in the targetNames tuple
                targets = getTargets(detections, frameNumber, targetNames=("car", "motorcycle", "bus", "truck"))

                # update track history
                updateHistory(trackedObjects, tracker, targets, db_connection, historyDepth=args.history, joblibdb=buffer2joblibTracks, k_velocity=args.k_velocity, k_acceleration=args.k_acceleration)

                # draw bounding boxes of filtered detections
                if args.show:
                    draw_boxes(trackedObjects, frame, colors, frameNumber)

                # load moving objects into buffer, that will be saved to the sqlite db at exit 
                for obj in trackedObjects:
                    if obj.isMoving:
                        # log to stdout
                        to_log.append(obj)
                        # save data in buffer
                        buffer2log.append([obj.objID, 
                                        obj.history[-1].frameID, 
                                        obj.history[-1].confidence, 
                                        obj.X, obj.Y, 
                                        obj.history[-1].Width, 
                                        obj.history[-1].Height,
                                        obj.VX, obj.VY, obj.AX, obj.AY, 
                                        obj.history_VX_calculated[-1], obj.history_VY_calculated[-1],
                                        obj.history_AX_calculated[-1], obj.history_AY_calculated[-1],
                                        obj.futureX, obj.futureY
                                        ])

                # show video frame
                if args.show:
                    cv.imshow("FRAME", cv.bitwise_or(frame, frame, mask=mask))

                # calculating fps from time before computation and time now
                fps = int(1/(time.time() - prev_time))

                # print FPS to stdout
                # print("FPS: {}".format(fps,))

                # print runtime logs
                log_to_stdout("FPS: {}".format(fps,), to_log[:])#, f"Number of moving objects: {num_of_moving_objs(trackedObjects)}", f"Number of objects: {len(trackedObjects)}", f"Buffersize: {len(buffer2log)}", f"Width {frameWidth} Height {frameHeight}")

                # press 'p' to pause playing the video
                if cv.waitKey(1) == ord('p'):
                    # press 'r' to resume
                    if cv.waitKey(0) == ord('r'):
                        continue
                # press 'q' to stop playing video
                if cv.waitKey(10) == ord('q'):
                    break
            except KeyboardInterrupt:
                break

        cap.release()
        # save trackedObjects into joblib database
        downscaled_tracks = downscale_TrackedObjects(buffer2joblibTracks, img) 
        dump(downscaled_tracks, path2joblib)
        print("Joblib database succesfully saved!")
        # log buffered detections in sqlite db
        databaseLogger.logBufferSpeedy(db_connection, img, buffer2log)
        databaseLogger.closeConnection(db_connection)

        if args.show:
            cv.destroyAllWindows()


    
    

if __name__ == "__main__":
    main()