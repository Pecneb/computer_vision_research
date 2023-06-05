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
from detector import getTargets
from detector import draw_boxes
from classifier import OneVSRestClassifierExtended
from masker import masker
from clustering import calc_cluster_centers

def parseArgs():
    """Handle command line arguments.

    Returns:
        args: arguments object, that contains the args given in the command line 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video.", type=str)
    parser.add_argument("--model", required=True, help="Path to trained model.", type=str)
    parser.add_argument("--history", type=int, default=30, 
                        help="How many detections will be saved in the track's history.")
    parser.add_argument("--max_cosine_distance",  type=float, default=10.0,
                        help="Gating threshold for cosine distance metric (object apperance)")
    parser.add_argument("--max_iou_distance", type=float, default=0.7)
    parser.add_argument("--nn_budget", type=float, default=100,
                        help="Maximum size of the apperance descriptors gallery. If None, no budget is enforced.")
    parser.add_argument("--feature_version", choices=['1', '2','3', '4', '7'], default='2', 
                        help="What version of feature vectors to use.")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of highest confidence predictions to show.")
    parser.add_argument("--record", action="store_true", default=False,
                        help="Use this flag if want to record video of prediction results.")
    parser.add_argument("--output", type=str, 
                        help="Use this flag if want to record video of prediction results.")
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
        fheight (int): Frame height of the video. 
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
    """Calculate the centroids of the area of interests, the gravity center of clusters.
    This function does not scales up the coordinates to the video's resolution.

    Args:
        tracks (list[TrackedObject]): List of TrackedObject objects.
        classes (list[int]): List of class labels ordered to the TrackedObjects.

    Returns:
        centroids: The centroids of the classes, returned in an np.ndarray. 
    """
    class_labels = np.array(list(set(classes)))
    centroids = np.zeros(shape=(class_labels.shape[0], 2))
    for l in class_labels:
        x_sum = 0
        y_sum = 0
        n = 0
        for t, c in zip(tracks, classes):
            if c == l:
                x_sum += t.history[-1].X
                y_sum += t.history[-1].Y
                n += 1
        centroids[l, 0] = x_sum/n
        centroids[l, 1] = y_sum/n
    return centroids

def upscale_aoi(centroids, framewidth: int, frameheight: int):
    """Scale centroids of clusters up to the video's resolution.

    Args:
        centroids (dict): Output of aoiextraction() function. 
        framewidth (int): Width resolution of the video. 
        frameheight (int): Height resolution of the video. 

    Returns:
        dict: Upscaled centroid coordinates. 
    """
    ratio = framewidth / frameheight
    retarr = centroids.copy()
    for i in range(centroids.shape[0]):
        retarr[i, 0] = centroids[i, 0] * framewidth / ratio
        retarr[i, 1] = centroids[i, 1] * frameheight
    return retarr 

def draw_prediction(trackedObject, centroid: list[np.ndarray], image: np.ndarray, framenum: int, predictions: np.ndarray, confidences: np.ndarray):
    """Draw prediction path.

    Args:
        detectionCoordinates (tuple): (x,y) Coordinates of the object. 
        centroid (np.ndarray): Cluster's centroid coordiantes.
        image (np.ndarray): Image to draw on. 
    """
    ratio = image.shape[1] / image.shape[0]
    X, Y = int(trackedObject.history[-1].X), int(trackedObject.history[-1].Y)
    W, H = int(trackedObject.history[-1].Width), int(trackedObject.history[-1].Height)
    if trackedObject.history[-1].frameID == framenum:
        bbox = (X, Y, W, H)
        left, top, right, bottom = bbox2points(bbox)
        cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 1)
        cv.putText(image, f"ID {trackedObject.objID} {predictions} {confidences[predictions[-1]]:3.2f}", (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        for i in range(centroid.shape[0]):
            if i == len(predictions)-1:
                cv.line(image, (X, Y), (int(centroid[i, 0]), int(centroid[i, 1])), (0,255,0), 3)
                cv.circle(image, (int(centroid[i, 0]), int(centroid[i, 1])), 10, (0,255,0), 3)
                cv.putText(image, f"Cluster: {predictions[-1]}", 
                            (int(centroid[i, 0]), int(centroid[i, 1])),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                cv.line(image, (X, Y), (int(centroid[i, 0]), int(centroid[i, 1])), (0,0,255), 3)
        

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

def prediction_paralell(t: TrackedObject, model: OneVSRestClassifierExtended, frame: np.ndarray, framewidth: int, frameheight: int, cluster_centroids: dict, cluster_centroids_upscaled: dict, feature_v3: bool = False):
    """Prediction algorithm for paralellism.

    Args:
        t (TrackedObject): _description_
        model (OneVSRestClassifierExtended): _description_
        frame (np.ndarray): _description_
        framewidth (int): _description_
        frameheight (int): _description_
        cluster_centroids (dict): _description_
        cluster_centroids_upscaled (dict): _description_
        feature_v3 (bool, optional): _description_. Defaults to False.
    """
    VERSION_3 = feature_v3
    feature = t.feature_(VERSION_3)
    if t.isMoving:
        if feature is not None:
            if VERSION_3:
                predictions = model.predict(np.array([t.downscale_feature(feature, framewidth, frameheight, VERSION_3)]), 1, centroids=cluster_centroids).reshape((-1))
            else:
                predictions = model.predict(np.array([t.downscale_feature(feature, framewidth, frameheight)]), 1).reshape((-1))
            #upscaledFeature = upscale_feature(featureVector=feature, framewidth=framewidth, frameheight=frameheight)
            centroids = [cluster_centroids_upscaled[p] for p in predictions]
            #draw_prediction((int(upscaledFeature[6]), int(upscaledFeature[7])), centroids, frame)
            draw_prediction((int(t.X), int(t.Y)), centroids, frame)

def main():
    args = parseArgs()

    VERSION_3 = int(args.feature_version) == 3

    from yolov7api import detect, COLORS

    cap = cv.VideoCapture(args.video)

    # create mask, so only in the area of interest will be used in detection 
    _, img = cap.read()
    mask = masker(img)

    framewidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frameheight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    if args.record:
        if args.output is None:
            print("No video output given exiting...")
            exit(0)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(filename=args.output, fourcc=fourcc, fps=30.00, frameSize=(int(framewidth), int(frameheight)), isColor=True)

    model = load_model(args.model)
    model.n_jobs = 18
    dataset = model.tracks
    """
    dataset = load_joblib_tracks(args.train_tracks) 
    if type(dataset[0]) != dict:
        raise TypeError("Bad type of dataset file. The train_tracks dataset should contain clustered data. List containing dicts with keys: track, class.")
    if args.all_tracks:
        all_tracks = load_joblib_tracks(args.all_tracks)

    dataset_filtered = [d for d in dataset if d["class"] != -1]
    """
    tracks = [d["track"] for d in dataset]
    classes = [d["class"] for d in dataset]

    # extract cluster centroids from dataset of classes and tracks
    cluster_centroids = cluster_centroids_upscaled(tracks, classes)
    # upscale the coordinates of the centroids to the video's scale
    cluster_centroids_upscaled = upscale_aoi(cluster_centroids, framewidth, frameheight)

    dsTracker = getTracker(initTrackerMetric(args.max_cosine_distance, args.nn_budget), historyDepth=args.history, max_iou_distance=args.max_iou_distance)

    history: list[TrackedObject] = [] # TrackedObjects

    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # create named window
    cv.namedWindow("Video")
    # create trackbar to be able to set frameposition
    setFrame = lambda frame_num: cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    cv.createTrackbar("Frame", "Video", 0, frame_count, setFrame)

    for frameidx in tqdm.tqdm(range(frame_count)):
        try:
            ret, frame = cap.read()
            if frame is None:
                print("Video enden, closing player.")
                break
            
            masked_frame = cv.bitwise_and(frame, frame, mask=mask)

            frameNum = cap.get(cv.CAP_PROP_POS_FRAMES)
            yoloDetections = detect(masked_frame) # get detections from yolo nn
            targetDetections = getTargets(yoloDetections, frameNum, targetNames=("car")) # get target detections and make Detection() objects
            updateHistory(history, dsTracker, targetDetections, historyDepth=args.history) # update track history and update tracker

            #draw_boxes(history, frame, COLORS, frameNum)

            for i in range(cluster_centroids_upscaled.shape[0]):
                cv.circle(frame, (int(cluster_centroids_upscaled[i][0]), int(cluster_centroids_upscaled[i][1])), 10, (0,0,255), 3)
                cv.putText(frame, f"Cluster: {i}", 
                            (int(cluster_centroids_upscaled[i][0]), int(cluster_centroids_upscaled[i][1])),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            for t in history:
                    if args.feature_version == '3':
                        feature = t.feature_v3()
                    if args.feature_version == '4':
                        feature = t.feature_v4()
                    if args.feature_version == '7':
                        feature = t.feature_v7()
                    else:
                        feature = t.feature_()
                    if t.isMoving:
                        if feature is not None:
                            if VERSION_3:
                                predictions = model.predict(np.array([t.downscale_feature(feature, framewidth, frameheight, VERSION_3)]), args.top_k, centroids=cluster_centroids).reshape((-1))
                            else:
                                predictions = model.predict(np.array([t.downscale_feature(feature, framewidth, frameheight)]), args.top_k).reshape((-1))
                                predictions_proba = model.predict_proba(np.array([t.downscale_feature(feature, framewidth, frameheight)])).reshape((-1))
                            centroids = np.array([cluster_centroids_upscaled[p] for p in predictions])
                            draw_prediction(t, centroids, frame, frameNum, predictions, predictions_proba)
            
            cv.imshow("Video", frame)
            cv.setTrackbarPos("Frame", "Video", int(frameNum))

            if args.record:
                out.write(frame)

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
    if args.record:
        out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()