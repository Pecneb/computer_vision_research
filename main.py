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
import hldnapi
import time

def parseArgs():
    """
    Read command line arguments
        input: video source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video.")
    args = parser.parse_args()
    return args.input

def main():
    input = parseArgs()
    try:
        # check, if input arg is a webcam
        input = int(input)
    except ValueError:
        print("Input source is a Video.")

    cap = cv.VideoCapture(input)
    if not cap.isOpened():
        print("Source cannot be opened.")
        exit(0)
    
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
    
        # time before computation
        prev_time = time.time()
    
        I, detections = hldnapi.detections2cvimg(frame)
    
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))
    
        cv.imshow("FRAME", I)
    
        print("FPS: {}".format(fps))

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()