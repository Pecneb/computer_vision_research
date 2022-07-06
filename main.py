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

import cv2 as cv
import argparse
import time
import numpy as np
from sklearn import linear_model
from historyClass import Detection, TrackedObject
from darknet import bbox2points, class_colors
from sklearn.metrics import euclidean_distances
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection as DeepSortDetection
from deep_sort.deep_sort.tracker import DeepSortTracker

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

def calcDist(prev, act):
    """Function to calculate distance between an object on previous frame and actual frame.

    Args:
        prev (Detection): Object from previous frame
        act (Detection): Object from actual frame

    Returns:
        xDist, yDist: distance of x coordiantes and distance of y coordiantes
    """
    xDist = abs(prev.X-act.X)
    yDist = abs(prev.Y-act.Y)
    return xDist, yDist

def updateHistory(detections, history, frameNumber, frameWidth, frameHeight, historyDepth=3, disThresh=0.05):
    """Function to update detection history

    Args:
        detections (list[Detection]): a list of new detection
        history (list[TrackedObject]): the tracking history
        frameNumber (int): number of the current video frame
        historyDepth (int): length of the history to be stored
        thresh (float, optional): Threshold to be able to tell if next obj is already detected or is a new one. Defaults to 0.1.
    """
    for next in detections:
        added = False
        for objHistory in history:
            try:
                prev = objHistory.history[-historyDepth]
            except:
                prev = objHistory.history[-1]
            xDist, yDist = calcDist(prev, next)
            if  (xDist < (prev.X * disThresh)) and (yDist < (prev.Y * disThresh)) and objHistory.label == next.label:
                objHistory.history.append(next)
                added = True
                # print(f"X threshold: {disThresh*prev.X}, Ythreshold: {disThresh*prev.Y}")
                # print(f"Thresh to movement, coord X: {(disThresh*0.1*prev.X)}, Y: {(disThresh*0.1*prev.Y)}")
                # the threshold for the non moving objects is still harcoded
                # TODO: find a good way to tell what objects are still or in motion
                if (xDist > (disThresh * prev.X * 0.25)) or (yDist > (disThresh * prev.Y * 0.25)):
                    print("ObjID: {} with xDist: {} and yDist: {} is moving".format(objHistory.objID, xDist, yDist))
                    objHistory.isMoving = True
                else:
                    print("ObjID: {} with xDist: {} and yDist: {} is not moving".format(objHistory.objID, xDist, yDist))
                    objHistory.isMoving = False
            # remove objects that are older than frameNumber-historyDepth
            if objHistory.history[-1].frameID < (frameNumber-historyDepth):
                try:
                    history.remove(objHistory)
                except:
                    continue
        if not added:
            history.append(TrackedObject(len(history)+1, next))

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
        #if detection.frameID == frameNumber:
        bbox = (detection.X, detection.Y, detection.Width, detection.Height)
        left, top, right, bottom = bbox2points(bbox)
        cv.rectangle(image, (left, top), (right, bottom), colors[detection.label], 1)
        cv.putText(image, "{} ID{} [{:.2f}]".format(detection.label, detections.objID, float(detection.confidence)),
                    (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[detection.label], 2)

def movementIsRight(X_hist, historyDepth):
    """Returns True if object moving left to right or False if right to left

    Args:
        X_hist (list(int)): list of X coordinates of length historyDepth
        historyDepth (int): the length of the x coord list

    Returns:
        bool: retval
    """
    try:
        return (X_hist[-1] > X_hist[-historyDepth])
    except:
        return (X_hist[-1] > X_hist[0])

def predictTraj(trackedObject, linear_model, historyDepth=3, futureDepth=30, image=None):
    """Calculating future trajectory of the trackedObject

    Args:
        trackedObject (TrackedObject): the tracked Object
        linear_model (sklearn linear_model): Linear model used to calculate the trajectory
        historyDepth (int, optional): the number of detections that the trajectory should be calculated from. Defaults to 3.
        futureDepth (int, optional): how far in the future should we predict. Defaults to 30.
        image (Opencv image, optional): if image is inputted, then trajectories are drawn to the image. Defaults to None.

    """
    X_train = np.array([det.X for det in trackedObject.history[-historyDepth-1:-1]])
    y_train = np.array([det.Y for det in trackedObject.history[-historyDepth-1:-1]])
    print(len(X_train), len(y_train))
    if len(X_train) == historyDepth and len(y_train) == historyDepth:
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        reg = linear_model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred
        if image is not None:
            for x,y in zip(trackedObject.futureX, trackedObject.futureY):
                cv.circle(image, (int(x),int(y)), 1, color=(0,0,255))

def draw_predictions(trackedObject, image, frameNumber):
    """Draw prediction information to image

    Args:
        trackedObject (TrackedObject): tracked object
        image (Opencv Image): image to draw on
        frameNumber (int): current number of the video frame
    """
    if trackedObject.history[-1].frameID == frameNumber:
        idx = 0
        for x, y in zip(trackedObject.futureX, trackedObject.futureY):
            if (idx % 4) == 0:
                cv.circle(image, (int(x), int(y)), 1, color=(0,0,255))
            idx += 1

def initMetric(max_cosine_distance, nn_budget, metric="cosine"):
    return nn_matching.NearestNeighborDistanceMetric(
        metric, max_cosine_distance, nn_budget)

def getTracker(metricObj):
    return DeepSortTracker(metricObj)

def makeDeepSortDetectionObject(x, y, w, h, confidence, label):
    return DeepSortDetection([(x-w/2), (y-h/2), w, h], float(confidence), [], labe, float(confidence), [], label)

# global var for adjusting stored history length
HISTORY_DEPTH = 3
FUTUREPRED = 30

def main():
    args = parseArgs()
    input = args.input
    if input is not None:
        import hldnapi
    # check input source
    try:
        input = int(input)
    except ValueError:
        print("Input source is a Video.")
    # get video capture object
    cap = cv.VideoCapture(input)
    # exit if video cant be opened
    if not cap.isOpened():
        print("Source cannot be opened.")
        exit(0)
    # forward declaration of history(list[TrackedObject])
    history = []
    # generating colors for bounding boxes based on the class names of the neural net
    colors = class_colors(hldnapi.class_names)
    # get frame width
    frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # get frame height
    frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    metric =
    # start main loop
    while(1):
        # get current frame from video
        ret, frame = cap.read()
        if frame is None:
            break
        # get current frame number
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)
        # time before computation
        prev_time = time.time()
        # use darknet neural net to detects objects
        detections = hldnapi.cvimg2detections(frame)
        # filter detections, only return the ones given in the targetNames tuple
        targets = getTargets(detections, frameNumber, targetNames=("person", "car"))
        # update track history
        updateHistory(targets, history, frameNumber, frameWidth, frameHeight, historyDepth=HISTORY_DEPTH)
        # draw bounding boxes of filtered detections
        draw_boxes(history, frame, colors, frameNumber)
        # run prediction algorithm and draw predictions on objects, that are in motion
        for obj in history:
            if obj.isMoving:
                predictTraj(obj, linear_model.RANSACRegressor(), historyDepth=HISTORY_DEPTH, futureDepth=FUTUREPRED)
                draw_predictions(obj, frame, frameNumber)
        # show video frame
        cv.imshow("FRAME", frame)
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))
        # print FPS to stdout
        print("FPS: {}".format(fps))
        # press 'q' to stop playing video
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
