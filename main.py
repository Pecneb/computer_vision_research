"""
    Predicting trajectories of objects
    Copyright (C) <2022>  Bence Peter

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

from tkinter import W
import cv2 as cv
import argparse
import time
import numpy as np
from historyClass import Detection, TrackedObject

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

def getTargets(detections, targetNames=['car, person']):
    """Function to extract detected objects from the detections, that labels are given in the targetNames argument

    Args:
        detections ((str, float, (int, int, int, int))): Return value of the darknet neuralnet
        targetNames (list, optional): List of labels, that should be tracked. Defaults to ['car', 'person'].

    Returns:
        targets ((str, float, (int, int, int, int))): Returns a list of Detection objects
    """
    targets = []
    for label, conf, bbox in detections:
        # bbox: x, y, w, h
        if label in targetNames:
            targets.append((label, conf, bbox))
    return targets 

def updateIMG(detections, IMG):
    """Function to draw a circle in the centre of all detected objects in the arg detections on the arg IMG cv image

    Args:
        detections ((str, float, (int, int, int, int))): _description_
        IMG (_type_): _description_
    """
    for obj in detections:
        cv.circle(IMG, (obj[2][0], obj[2][1]), 2, (0,0,255), 2)

def calcDist(prev, act):
    """Function to calculate distance between an object on previous frame and actual frame.

    Args:
        prev (tuple(label, confidence, bbox(x,y,w,h))): Object from previous frame
        act (tuple(label, confidence, bbox(x,y,w,h))): Object from actual frame

    Returns:
        xDist, yDist: distance of x coordiantes and distance of y coordiantes
    """
    xDist = abs(prev[2][0]-act[2][0])
    yDist = abs(prev[2][1]-act[2][1])
    return xDist, yDist

def updateHistory(detections, history, thresh=0.05):
    """Function to update object history

    Args:
        detections (list of (label, confidence, bbox)): output of darknet
        history (2D matrix): objects's history, each row is a history of an object
        thresh (float, optional): The percentage of max distance. Defaults to 0.05.
    """
    for new in detections:
        added = False
        for objHistory in history:
            xDist, yDist = calcDist(objHistory[-1], new)
            if xDist < (objHistory[-1][2][0]*thresh) and yDist < (objHistory[-1][2][0]*thresh):
                objHistory.append(new)
                added = True
        if not added:
            history.append([new])

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

    # imgMask = np.zeros_like(I, dtype=np.uint8)

    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
    
        # time before computation
        prev_time = time.time()
    
        I, detections = hldnapi.detections2cvimg(frame)
        
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))
        
        targets = getTargets(detections, targetNames=("person", "car"))

        updateHistory(targets, history)

        # printDetections(targets)

        updateIMG(targets, I)

        # cv.circle(I, (history[3][-1][2][0],history[3][-1][2][1]), 2, (255,0,0), 2)

        # imgToShow = cv.add(imgMask, I)

        cv.imshow("FRAME", I)
    
        print("FPS: {}".format(fps))

        # print(history[1][-2:-1])

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()