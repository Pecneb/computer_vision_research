"""
    Predicting trajectories of objects
    Copyright (C) 2022 Bence Peter

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
    """Function for printing out detections

    Args:
        detections (tuple(label, confidence, bbox(x,y,w,h))): output of darknet detector
    """
    for obj in detections:
        # bbox: x, y, w, h
        print("Class: {}, Confidence: {}, Position: {}".format(obj[0], obj[1], obj[2]))

def getTargets(detections, targetNames=[]):
    """Function for extracting targets from detections

    Args:
        detections (tuple(label, confidence, bbox(x,y,w,h))): output of darknet detector
        targetNames (list, optional): list of labels, that should be extracted from detections. Defaults to [].

    Returns:
        list: list of detections with the labels of targetNames
    """
    targets = []
    for label, conf, bbox in detections:
        # bbox: x, y, w, h
        if label in targetNames:
            targets.append((label, conf, bbox))
    return targets 

def updateIMG(detections, IMG):
    """Functions for drawing a circle in the center of detected objects 

    Args:
        detections (tuple(label, confidence, bbox(x,y,w,h))): output of darknet detector or getTargets() function
        IMG (OpenCV IMG object): input image
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

    # only initialize darknet when theres valid input video
    if input is not None:
        import hldnapi

    try:
        # if input source is number, then it is a webcam 
        input = int(input)
    except ValueError:
        print("Input source is a Video.")

    cap = cv.VideoCapture(input)
    if not cap.isOpened():
        print("Video cannot be opened.")
        exit(0) 

    # defining history list for later usage
    # 2D MTX: rows are the objects and columns are the timeline
    # obj1 1. position 2. position ...
    # obj2 1. pos 2. pos ...
    history = []

    # playing video frame by frame until 'q' is pressed or video ends
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
    
        # time before computation
        prev_time = time.time()
    
        # using darknet on next video frame
        I, detections = hldnapi.detections2cvimg(frame)
        
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))

        # extracting only the objects with the inputted label
        targets = getTargets(detections, targetNames=("person", "car"))

        # update history for tracking objects
        updateHistory(targets, history)

        # printDetections(targets)

        # draw red dots on selected objects center for testing purposes
        updateIMG(targets, I)

        # marking an object with blue dot for testing purposes
        cv.circle(I, (history[1][-1][2][0],history[1][-1][2][1]), 2, (255,0,0), 2)

        cv.imshow("FRAME", I)

        # printing out fps to stdout
        print("FPS: {}".format(fps))

        # printing out the length of the longest history for testing purposes
        longestHist = history[0]
        for objHistory in history:
            if len(objHistory) > len(longestHist):
                longestHist = objHistory
                print(len(longestHist))

        # press 'q' to quit
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()