"""
    Visualize classification results
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
from classifier import BinaryClassifier
import cv2 as cv
import argparse
import tqdm

def parseArgs():
    """Handle command line arguments.

    Returns:
        args: arguments object, that contains the args given in the command line 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video.", type=str)
    parser.add_argument("--model", required=True, help="Path to trained model.", type=str)
    
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()

    cap = cv.VideoCapture(args.input)

    framewidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frameheight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    for frameidx in tqdm.tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
        try:
            ret, frame = cap.read()
            if frame is None:
                print("Video enden, closing player.")
                break


            pass
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()