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

from ipaddress import collapse_addresses
from unicodedata import name
import cv2 as cv
import argparse
import time
import numpy as np
from historyClass import Detection, TrackedObject
from darknet import bbox2points, class_colors

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

def updateHistory(detections, history, tresh=0.05):
    """Function to update detection history

    Args:
        detections (list[Detection]): a list of new detection
        history (list[TrackedObject]): the tracking history
        tresh (float, optional): Treshold to be able to tell if next obj is already detected or is a new one. Defaults to 0.05.
    """
    for next in detections:
        added = False
        for objHistory in history:
            last = objHistory.history[-1]
            xDist, yDist = calcDist(last, next)
            if xDist < (last.X*tresh) and yDist < (last.X*tresh) and objHistory.label == next.label:
                objHistory.history.append(next)
                added = True
        if not added:
            history.append(TrackedObject(len(history)+1, next))

def draw_boxes(history, image, colors, frameNumber):
    for detections in history:
        detection = detections.history[-1]
        if detection.frameID == frameNumber:
            bbox = (detection.X, detection.Y, detection.Width, detection.Height)
            left, top, right, bottom = bbox2points(bbox)
            cv.rectangle(image, (left, top), (right, bottom), colors[detection.label], 1)
            cv.putText(image, "{} ID{} [{:.2f}]".format(detection.label, detections.objID, float(detection.confidence)),
                        (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[detection.label], 2)
    return image

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

    history = []
    colors = class_colors(hldnapi.class_names)

    # imgMask = np.zeros_like(I, dtype=np.uint8)

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