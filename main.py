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

import cv2 as cv
import argparse
import time
import numpy as np

def parseArgs():
    """
    Read command line arguments
        input: video source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video.")
    args = parser.parse_args()
    return args.input

def printDetections(detections):
    """
    Print labels, confidences and bounding boxes positions of detections
    """
    for obj in detections:
        # bbox: x, y, w, h
        print("Class: {}, Confidence: {}, Position: {}".format(obj[0], obj[1], obj[2]))

def getTargets(detections, targetNames=[]):
    """
    Retrieve the positions of targeted objects.
    detections: output of hldnapi.detections2cvimg
    targetNames: labels of object to be retrieved
    """
    targets = []
    for label, conf, bbox in detections:
        # bbox: x, y, w, h
        if label in targetNames:
            targets.append((label, conf, bbox))
    return targets 

def drawCenters(detections, img):
    for obj in detections:
        cv.circle(img, (obj[2][0], obj[2][1]), 2, (0,0,255), 2)

def calcDist(prev, act):
    xDist = abs(prev[2][0]-act[2][0])
    yDist = abs(prev[2][1]-act[2][1])
    return xDist, yDist

def updateHistory(detections, history, thresh=0.05):
    """
    updates the history of objects
    if the old obj center's and new obj center's distance is within the threshold,
    then add the new one to the old one's history,
    but if no close centerpoint is found, make new history for the obj
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

        drawCenters(targets, I)

        cv.circle(I, (history[3][-1][2][0],history[3][-1][2][1]), 2, (255,0,0), 2)

        cv.imshow("FRAME", I)
    
        print("FPS: {}".format(fps))

        # print(history[1][-2:-1])

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()