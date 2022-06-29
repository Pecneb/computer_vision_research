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

def parseArgs():
    """Function for Parsing Arguments

    Returns:
        args.input: input video source given in command line argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video.")
    args = parser.parse_args()
    return args.input

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

def updateHistory(detections, history, disThresh=0.05):
    """Function to update detection history

    Args:
        detections (list[Detection]): a list of new detection
        history (list[TrackedObject]): the tracking history
        thresh (float, optional): Threshold to be able to tell if next obj is already detected or is a new one. Defaults to 0.05.
    """
    for next in detections:
        added = False
        for objHistory in history:
            last = objHistory.history[-1]
            # xDist, yDist = calcDist(last, next)
            euclidean = euclidean_distances([[last.X, last.Y]], [[next.X, next.Y]])
            if (euclidean < (last.X*disThresh) and euclidean < (last.Y*disThresh)) and objHistory.label == next.label:
                objHistory.history.append(next)
                added = True
                if euclidean > 1.5: 
                    # print("x coord distance: {}, y coord distance: {}".format(xDist, yDist))
                    print("Euclidean distance: {}".format(euclidean))
                    objHistory.isMoving = True
                else:
                    objHistory.isMoving = False
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
        if detection.frameID == frameNumber:
            bbox = (detection.X, detection.Y, detection.Width, detection.Height)
            left, top, right, bottom = bbox2points(bbox)
            cv.rectangle(image, (left, top), (right, bottom), colors[detection.label], 1)
            cv.putText(image, "{} ID{} [{:.2f}]".format(detection.label, detections.objID, float(detection.confidence)),
                        (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[detection.label], 2)
            cv.circle(image, (detection.X, detection.Y), 1, color=(0,255,0))

def movementIsRight(X_hist, historyDepth):
    return (X_hist[-1] > X_hist[-historyDepth])

def predictTraj(trackedObject, linear_model, historyDepth=3, futureDepth=30, image=None):
    X_train = np.array([det.X for det in trackedObject.history[-historyDepth-1:-1]])
    y_train = np.array([det.Y for det in trackedObject.history[-historyDepth-1:-1]])
    if len(X_train) >= historyDepth and len(y_train) >= historyDepth:
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        reg = linear_model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred[0]
        if image is not None:
            for x,y in zip(X_test, y_pred):
                cv.circle(image, (int(x),int(y[0])), 1, color=(0,0,255))
    else:
        return False

def main():
    input = parseArgs()
    if input is not None:
        import hldnapi

    try:
        # check, if input arg is a webcam
        input = int(input)
    except ValueError:
        print("Input source is a Video.")

    cap = cv.VideoCapture(input)
    if not cap.isOpened():
        print("Source cannot be opened.")
        exit(0) 

    # forward declaration of history(list[TrackedObject])
    history = []
    # generating colors for bounding boxes based on the class names of the neural net
    colors = class_colors(hldnapi.class_names)

    while(1):
        ret, frame = cap.read()
        if frame is None:
            break

        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES) 

        # time before computation
        prev_time = time.time()

        # use darknet neural net to detects objects 
        detections = hldnapi.cvimg2detections(frame)
    
        targets = getTargets(detections, frameNumber, targetNames=("person", "car"))

        updateHistory(targets, history)

        draw_boxes(history, frame, colors, frameNumber)

        for obj in history:
            if obj.isMoving:
                predictTraj(obj, linear_model.LinearRegression(), historyDepth=10, futureDepth=100, image=frame)

        cv.imshow("FRAME", frame)
        
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))
        
        print("FPS: {}".format(fps))

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()