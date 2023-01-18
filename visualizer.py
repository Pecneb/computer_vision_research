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
from processing_utils import load_joblib_tracks, load_model
from dataManagementClasses import Detection, TrackedObject
import cv2 as cv
import argparse
import tqdm
import numpy as np
from deepsortTracking import initTrackerMetric, getTracker, updateHistory
from main import getTargets
from main import draw_boxes

def parseArgs():
    """Handle command line arguments.

    Returns:
        args: arguments object, that contains the args given in the command line 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video.", type=str)
    parser.add_argument("--model", required=True, help="Path to trained model.", type=str)
    parser.add_argument("--train_tracks", required=True, help="Path to the tracks joblib file.", type=str)
    parser.add_argument("--all_tracks", help="Not only the tracks used for model training, rather all detections.", type=str)
    parser.add_argument("--history", help="How many detections will be saved in the track's history.", type=int, default=30)
    parser.add_argument("--max_cosine_distance", help="Gating threshold for cosine distance metric (object apperance)", type=float, default=10.0)
    parser.add_argument("--nn_budget", help="Maximum size of the apperance descriptors gallery. If None, no budget is enforced.", type=float, default=100)
    parser.add_argument("--feature_version", help="What version of feature vectors to use.", choices=[2,3], default=2)
    
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

def drawbbox(bbox: tuple, image: np.ndarray):
    """Draw bounding box of an object to the given image.

    Args:
        detection (Detection): Detection object. 
        image (np.ndarray): OpenCV image object.
    """
    bboxUpscaled = upscalebbox(bbox, image.shape[1], image.shape[0])
    left, top, right, bottom = bbox2points(bboxUpscaled)
    cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 1)

def aoiextraction(tracks, classes):
    """Calculate the centroids of the are of interests, the gravity center of clusters.
    This function does not scales up the coordinates to the video's resolution.

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
    """Scale centroids of clusters up to the video's resolution.

    Args:
        centroids (dict): Output of aoiextraction() function. 
        framewidth (int): Width resolution of the video. 
        frameheight (int): Height resolution of the video. 

    Returns:
        dict: Upscaled centroid coordinates. 
    """
    ratio = framewidth / frameheight
    ret = {}
    for c in centroids:
        ret[c] = np.array([centroids[c][0] * framewidth / ratio, centroids[c][1] * frameheight])
    return ret

def draw_prediction(detectionCoordinates: tuple, centroid: list[np.ndarray], image: np.ndarray):
    """Draw prediction path.

    Args:
        detectionCoordinates (tuple): (x,y) Coordinates of the object. 
        centroid (np.ndarray): Cluster's centroid coordiantes.
        image (np.ndarray): Image to draw on. 
    """
    ratio = image.shape[1] / image.shape[0]
    X, Y = detectionCoordinates
    for c in centroid:
        cv.line(image, (X, Y), (int(c[0]), int(c[1])), (0,0,255), 3)

def upscale_coordinates(p1, p2, image: np.ndarray):
    ratio = image.shape[1]/image.shape[0]
    return (p1 * image.shape[1] / ratio), (p2 * image.shape[0])

def feature_at_idx(track: TrackedObject, frameidx: int):
    lastidx = 0
    for d in track.history:
        if d.frameID == frameidx:
            break
        lastidx += 1
    if lastidx == len(track.history):
        return None 
    x_0 = track.history[0].X
    y_0 = track.history[0].Y
    vx_0 = track.history[0].VX
    vy_0 = track.history[0].VY
    x_1 = track.history[lastidx//2].X
    y_1 = track.history[lastidx//2].Y
    x_2 = track.history[lastidx].X
    y_2 = track.history[lastidx].Y
    vx_2 = track.history[lastidx].VX
    vy_2 = track.history[lastidx].VY
    return np.array([x_0, y_0, vx_0, vy_0, x_1, y_1, x_2, y_2, vx_2, vy_2])

def upscale_feature(featureVector: np.ndarray, framewidth: int, frameheight: int):
        """Rescale normalized coordinates with the given frame sizes.

        Args:
            featureVector (np.ndarray): Feature vector of the track.
            framewidth/ratio (int): Width of the video frame. 
            frameheight (int): Height of the video frame. 

        Returns:
            np.ndarray: upscaled feature vector
        """
        ratio = framewidth / frameheight
        return np.array([framewidth/ratio*featureVector[0], frameheight*featureVector[1], framewidth/ratio*featureVector[2],
                        frameheight*featureVector[3], framewidth/ratio*featureVector[4], frameheight*featureVector[5],
                        framewidth/ratio*featureVector[6], frameheight*featureVector[7], framewidth/ratio*featureVector[8],
                        frameheight*featureVector[9]])

def main():
    args = parseArgs()

    VERSION_3 = args.feature_version == 3

    from yolov7api import detect, COLORS

    cap = cv.VideoCapture(args.video)

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # create named window
    cv.namedWindow("Video")
    # create trackbar to be able to set frameposition
    setFrame = lambda frame_num: cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    cv.createTrackbar("Frame", "Video", 0, frame_count, setFrame)

    framewidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frameheight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    model = load_model(args.model)
    dataset = load_joblib_tracks(args.train_tracks) 
    if args.all_tracks:
        all_tracks = load_joblib_tracks(args.all_tracks)

    dataset_filtered = [d for d in dataset if d["class"] != -1]
    tracks = [d["track"] for d in dataset_filtered]
    classes = [d["class"] for d in dataset_filtered]

    # extract cluster centroids from dataset of classes and tracks
    cluster_centroids = aoiextraction(tracks, classes)
    # upscale the coordinates of the centroids to the video's scale
    cluster_centroids_upscaled = upscale_aoi(cluster_centroids, framewidth, frameheight)

    dsTracker = getTracker(initTrackerMetric(args.max_cosine_distance, args.nn_budget), historyDepth=args.history)

    history = [] # TrackedObjects

    for frameidx in tqdm.tqdm(range(frame_count)):
        try:
            ret, frame = cap.read()
            if frame is None:
                print("Video enden, closing player.")
                break

            frameNum = cap.get(cv.CAP_PROP_POS_FRAMES)
            yoloDetections = detect(frame) # get detections from yolo nn
            targetDetections = getTargets(yoloDetections, frameNum, targetNames=("car")) # get target detections and make Detection() objects
            updateHistory(history, dsTracker, targetDetections, historyDepth=args.history) # update track history and update tracker

            draw_boxes(history, frame, COLORS, frameNum)

            #if args.all_tracks:
            #    for t in all_tracks:
            #        for d in t.history:
            #            if d.frameID == frameidx:
            #                drawbbox(d, frame)
            #else:
            #    for t in tracks:
            #        for d in t.history:
            #            if d.frameID == frameidx:
            #                drawbbox(d, frame)

            for c in cluster_centroids_upscaled:
                cv.circle(frame, (int(cluster_centroids_upscaled[c][0]), int(cluster_centroids_upscaled[c][1])), 10, (0,0,255), 3)
                if args.all_tracks:
                    for t in all_tracks:
                            feature = feature_at_idx(t, frameidx)
                            if feature is not None:
                                predictions = model.predict(np.array([feature]))
                                upscaledFeature = upscale_feature(featureVector=feature, framewidth=framewidth, frameheight=frameheight)
                                centroids = [cluster_centroids_upscaled[p[0]] for p in predictions]
                                draw_prediction((int(upscaledFeature[6]), int(upscaledFeature[7])), centroids, frame)
                else:
                    for t in history:
                            # feature = feature_at_idx(t, frameidx)
                            feature = t.feature_(VERSION_3)
                            if t.isMoving:
                                if feature is not None:
                                    if VERSION_3:
                                        predictions = model.predict(np.array([t.downscale_feature(feature, framewidth, frameheight)]), 1, centroids=cluster_centroids).reshape((-1))
                                    else:
                                        predictions = model.predict(np.array([t.downscale_feature(feature, framewidth, frameheight)]), 1).reshape((-1))
                                    #upscaledFeature = upscale_feature(featureVector=feature, framewidth=framewidth, frameheight=frameheight)
                                    centroids = [cluster_centroids_upscaled[p] for p in predictions]
                                    #draw_prediction((int(upscaledFeature[6]), int(upscaledFeature[7])), centroids, frame)
                                    draw_prediction((int(feature[6]), int(feature[7])), centroids, frame)
                    #for d in t.history:
                        #if d.frameID == frameidx:
                        #    draw_prediction((int(d.X * framewidth / (framewidth/frameheight)), int(d.Y * frameheight)), cluster_centroids_upscaled[l], frame)
            #for t in tracks:
            #    cv.circle(frame, (int(t.history[-1].X * framewidth / (framewidth/frameheight)), int(t.history[-1].Y * frameheight)), 1, (0,0,255), -1)

            cv.imshow("Video", frame)
            cv.setTrackbarPos("Frame", "Video", int(frameNum))

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
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()