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
import warnings
warnings.warn = warn

import cv2 as cv
import argparse
import time
import numpy as np
from historyClass import Detection
from deepsortTracking import initTrackerMetric, getTracker, updateHistory
from predict import draw_history, draw_predictions, predictLinPoly, predictWeightedLinPoly 
import databaseLogger
import os
import tqdm

def parseArgs():
    """Function for Parsing Arguments

    Returns:
        args.input: input video source given in command line argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video.")
    parser.add_argument("--max_cosine_distance", type=float, default=10.0,
                        help="remove detections with confidence below this value")
    parser.add_argument("--nn_budget", type=float, default=100,
                        help="remove detections with confidence below this value")
    parser.add_argument("--min_detection_height", type=float, default=0,
                        help="remove detections with confidence below this value")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                        help="remove detections with confidence below this value")
    parser.add_argument("--nms_max_overlap", type=float, default=1.0,
                        help="remove detections with confidence below this value")
    parser.add_argument("-d", "--device", default='cuda', help="Choose device to run neuralnet. example: cpu, cuda, 0,1,2,3...")
    parser.add_argument("--yolov7", default=1, type=int, help="Choose which yolo model to use. Choices: yolov7 = 1, yolov4 = 0")
    parser.add_argument("--resume", "-r", action="store_true", help="Use this flag if want to resume video from last session left off.")
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
            cv.putText(image, "{} ID{} [{:.2f}]".format(detection.label, detections.objID, float(detection.confidence)),
                        (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[detection.label], 2)

def log_to_stdout(*args):
    print('\n' * 200) 
    for arg in args:
        if type(arg) is list:
            for ar in arg:
                print(ar)
        else:
            print(arg)
    print('\n'*5)

# global var for adjusting stored history length
HISTORY_DEPTH = 30 
FUTUREPRED = 30 

def main():
    args = parseArgs()
    if args.yolov7:
        import torch
        
        import yolov7api 
    else:
        import hldnapi
        from darknet import class_colors
    input = args.input
    # check input source
    try:
        input = int(input)
        vidname = time.strftime("%Y%m%d_%H%M%S") # if input is camera source, then db name is the datetime
        db_name = vidname + ".db"
        databaseLogger.init_db(vidname, db_name)
    except ValueError:
        print("Input source is a Video.")
        # extracting video name from input source path and creating database name from it
        vidname = input.split('/', )[-1]
        vidname = vidname.split('.')[0]
        db_name = vidname + ".db"
        databaseLogger.init_db(vidname) # initialize database for logging
    # get video capture object
    cap = cv.VideoCapture(input)
    # exit if video cant be opened
    if not cap.isOpened():
        print("Source cannot be opened.")
        exit(0)
    # forward declaration of history(list[TrackedObject])
    history = []
    # generating colors for bounding boxes based on the class names of the neural net
    if args.yolov7:
        colors = yolov7api.COLORS 
    else:
        colors = hldnapi.colors
    # get frame width
    frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # get frame height
    frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    # create DeepSortTracker with command line arguments
    tracker = getTracker(initTrackerMetric(args.max_cosine_distance, args.nn_budget), historyDepth=HISTORY_DEPTH)
    # create database connection
    db_connection = databaseLogger.getConnection(os.path.join("research_data", vidname, db_name))
    # log metadata to database
    if args.yolov7:
        yoloVersion = '7'
    else:
        yoloVersion = '4'
    databaseLogger.logMetaData(db_connection, HISTORY_DEPTH, FUTUREPRED, yoloVersion, yolov7api.DEVICE, yolov7api.IMGSZ, yolov7api.STRIDE, yolov7api.CONF_THRES, yolov7api.IOU_THRES)
    # resume video where it was left off, if resume flag is set
    if args.resume:
        lastframeNum = databaseLogger.getLatestFrame(db_connection)
        if lastframeNum > 0 and lastframeNum < cap.get(cv.CAP_PROP_FRAME_COUNT):
            cap.set(cv.CAP_PROP_POS_FRAMES, lastframeNum-1) 
    else:
        lastframeNum = 0
    # start main loop
    for frameIDX in tqdm.tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))), initial=lastframeNum):
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
        # use darknet neural net to detects objects
        if args.yolov7:
            detections = yolov7api.detect(frame) 
        else:
            detections = hldnapi.cvimg2detections(frame)
        # filter detections, only return the ones given in the targetNames tuple
        targets = getTargets(detections, frameNumber, targetNames=("person", "car"))
        # update track history
        updateHistory(history, tracker, targets, db_connection, historyDepth=HISTORY_DEPTH)
        # draw bounding boxes of filtered detections
        draw_boxes(history, frame, colors, frameNumber)
        # run prediction algorithm and draw predictions on objects, that are in motion
        for obj in history:
            if obj.isMoving:
                # calculate predictions
                predictWeightedLinPoly(obj, historyDepth=HISTORY_DEPTH, futureDepth=FUTUREPRED)
                # draw predictions and tracking history
                draw_predictions(obj, frame, frameNumber)
                draw_history(obj, frame, frameNumber)
                # log to stdout
                to_log.append(obj)
                # log to database
                if not databaseLogger.entryExists(db_connection, obj.objID, obj.history[-1].frameID):
                    databaseLogger.logDetection(db_connection, 
                                            frame, 
                                            obj.objID, 
                                            obj.history[-1].frameID, 
                                            obj.history[-1].confidence, 
                                            obj.X, obj.Y, 
                                            obj.history[-1].Width, 
                                            obj.history[-1].Height,
                                            obj.VX, obj.VY)
        # show video frame
        cv.imshow("FRAME", frame)
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))
        # print FPS to stdout
        # print("FPS: {}".format(fps,))
        log_to_stdout("FPS: {}".format(fps,), to_log[:])
        # press 'p' to pause playing the video
        if cv.waitKey(1) == ord('p'):
            # press 'r' to resume
            if cv.waitKey(0) == ord('r'):
                continue
        # press 'q' to stop playing video
        if cv.waitKey(10) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    databaseLogger.closeConnection(db_connection)

if __name__ == "__main__":
    main()