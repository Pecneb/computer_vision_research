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
from classifier import OneVSRestClassifierExtended 
from processing_utils import load_joblib_tracks, load_model
from dataManagementClasses import Detection, TrackedObject
import cv2 as cv
import argparse
import tqdm
import numpy as np

def parseArgs():
    """Handle command line arguments.

    Returns:
        args: arguments object, that contains the args given in the command line 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video.", type=str)
    parser.add_argument("--model", required=True, help="Path to trained model.", type=str)
    parser.add_argument("--tracks", required=True, help="Path to the tracks joblib file.", type=str)
    parser.add_argument("--all_tracks", help="Not only the tracks used for model training, rather all detections.", type=str)
    
    args = parser.parse_args()
    return args

def upscalebbox(bbox, fwidth, fheight):
    """Upscale normalized coordinates to the video's frame size.
    The downscaling method was: X = X * fwidth / (fwidth/fheight), 
                                Y = Y * fheight
                                W = W * fwidth / (fwidth/fheight), 
                                H = H * fheight

    Args:
        bbox (tuple): Tuple of 4 values (X,Y,W,H) 
        fwidth (int): Frame width of the video. 
        fheight (_type_): Frame height of the video. 
    """
    ratio = fwidth / fheight
    X, Y, W, H = bbox
    X = (X * fwidth) / ratio 
    W = (W * fwidth) / ratio
    Y = Y * fheight 
    H = H * fheight 
    return X, Y, W, H

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

def drawbbox(detection: Detection, image: np.ndarray):
    """Draw bounding box of an object to the given image.

    Args:
        detection (Detection): Detection object. 
        image (np.ndarray): OpenCV image object.
    """
    bbox = (detection.X, detection.Y, detection.Width, detection.Height)
    bbox = upscalebbox(bbox, image.shape[1], image.shape[0])
    left, top, right, bottom = bbox2points(bbox)
    cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 1)

def aoiextraction(tracks, classes):
    """Calculate the centroids of the are of interests, the gravity center of clusters.

    Args:
        tracks (list[TrackedObject]): List of TrackedObject objects.
        classes (list[int]): List of class labels ordered to the TrackedObjects.

    Returns:
        dict: The centroids of the classes, returned in a dictionary format. 
    """
    class_labels = np.array(list(set(classes)))
    centroids = {}
    for l in class_labels:
        x_sum = 0
        y_sum = 0
        n = 0
        for t, c in zip(tracks, classes):
            if c == l:
                x_sum += t.history[-1].X
                y_sum += t.history[-1].Y
                n += 1
        centroids[l] = np.array([x_sum/n, y_sum/n])
    return centroids

def upscale_aoi(centroids: dict, framewidth: int, frameheight: int):
    ratio = framewidth / frameheight
    ret = {}
    for c in centroids:
        ret[c] = np.array([centroids[c][0] * framewidth / ratio, centroids[c][1] * frameheight / ratio])
    return ret

def main():
    args = parseArgs()

    cap = cv.VideoCapture(args.video)

    framewidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frameheight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    model = load_model(args.model)
    dataset = load_joblib_tracks(args.tracks) 
    if args.all_tracks:
        all_tracks = load_joblib_tracks(args.all_tracks)

    dataset_filtered = [d for d in dataset if d["class"] != -1]
    tracks = [d["track"] for d in dataset_filtered]
    classes = [d["class"] for d in dataset_filtered]

    cluster_centroids = aoiextraction(tracks, classes)
    cluster_centroids_upscaled = upscale_aoi(cluster_centroids, framewidth, frameheight)

    for frameidx in tqdm.tqdm(range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))):
        try:
            ret, frame = cap.read()
            if frame is None:
                print("Video enden, closing player.")
                break

            if args.all_tracks:
                for t in all_tracks:
                    for d in t.history:
                        if d.frameID == frameidx:
                            drawbbox(d, frame)
            else:
                for t in tracks:
                    for d in t.history:
                        if d.frameID == frameidx:
                            drawbbox(d, frame)

            #for c in cluster_centroids_upscaled:
            #    cv.circle(frame, (int(cluster_centroids_upscaled[c][0]), int(cluster_centroids_upscaled[c][1])), 50, (0,0,255), -1)
            for t in tracks:
                cv.circle(frame, (int(t.history[-1].X * framewidth / (framewidth/frameheight)), int(t.history[-1].Y * frameheight)), 1, (0,0,255), -1)

            cv.imshow("Frame", frame)

            # pause video
            if cv.waitKey(1) == ord('p'):
                # resume video
                key = cv.waitKey(0)
                if key == ord('r'):
                    continue
                elif key == ord('q'):
                    break 
            # quit vudeo player
            if cv.waitKey(1) == ord('q'):
                break
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()