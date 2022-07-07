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
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection as DeepSortDetection
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker

def parseArgs():
    """Function for Parsing Arguments

    Returns:
        args.input: input video source given in command line argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to video.")
    parser.add_argument("--max_cosine_distance", type=float, default=10.0,
                        help="remove detections with confidence below this value")
    parser.add_argument("--nn_budget", type=float, default=100,
                        help="remove detections with confidence below this value")
    parser.add_argument("--min_detection_height", type=float, default=0,
                        help="remove detections with confidence below this value")
    parser.add_argument("--min_confidence", type=float, default=0.7,
                        help="remove detections with confidence below this value")
    parser.add_argument("--nms_max_overlap", type=float, default=1.0,
                        help="remove detections with confidence below this value")
    args = parser.parse_args()
    return args

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

def movementIsRight(X_hist, historyDepth):
    """Returns True if object moving left to right or False if right to left

    Args:
        X_hist (list(int)): list of X coordinates of length historyDepth
        historyDepth (int): the length of the x coord list

    Returns:
        bool: retval
    """
    try:
        return (X_hist[-1] > X_hist[-historyDepth])
    except:
        return (X_hist[-1] > X_hist[0])

def predictTraj(trackedObject, linear_model, historyDepth=3, futureDepth=30, image=None):
    """Calculating future trajectory of the trackedObject

    Args:
        trackedObject (TrackedObject): the tracked Object
        linear_model (sklearn linear_model): Linear model used to calculate the trajectory
        historyDepth (int, optional): the number of detections that the trajectory should be calculated from. Defaults to 3.
        futureDepth (int, optional): how far in the future should we predict. Defaults to 30.
        image (Opencv image, optional): if image is inputted, then trajectories are drawn to the image. Defaults to None.

    """
    X_train = np.array([det.X for det in trackedObject.history[-historyDepth-1:-1]])
    y_train = np.array([det.Y for det in trackedObject.history[-historyDepth-1:-1]])
    if len(X_train) >= 3 and len(y_train) >= 3:
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        reg = linear_model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred
        if image is not None:
            for x,y in zip(trackedObject.futureX, trackedObject.futureY):
                cv.circle(image, (int(x),int(y)), 1, color=(0,0,255))

def draw_predictions(trackedObject, image, frameNumber):
    """Draw prediction information to image

    Args:
        trackedObject (TrackedObject): tracked object
        image (Opencv Image): image to draw on
        frameNumber (int): current number of the video frame
    """
    if trackedObject.history[-1].frameID == frameNumber:
        idx = 0
        for x, y in zip(trackedObject.futureX, trackedObject.futureY):
            if (idx % 4) == 0:
                cv.circle(image, (int(x), int(y)), 1, color=(0,0,255))
            idx += 1

def initDeepSortTrackerMetric(max_cosine_distance, nn_budget, metric="cosine"):
    """DeepSort metric factory

    Args:
        max_cosine_distance (float): Gating threshold for cosine distance metric (object appearance) 
        nn_budget (int): Maximum size of the appearance descriptor gallery. If None, no budget is enforced. 
        metric (str, optional): Distance metric type. Defaults to "cosine".

    Returns:
        metric: NearestNeighborDistanceMetric 
    """
    return nn_matching.NearestNeighborDistanceMetric(
        metric, max_cosine_distance, nn_budget)

def getDeepSortTracker(metricObj, historyDepth):
    """DeepSort Tracker object fractory

    Args:
        metricObj (metric): DistanceMetric object for Tracker object from deep_sort.deep_sort.tracker.Tracker class 

    Returns:
        tracker: deep_sort Tracker object 
    """
    return DeepSortTracker(metricObj, historyDepth)

def makeDeepSortDetectionObject(darknetDetection):
    """DeepSort Detection object factory

    Args:
        darknetDetection (Detection): Detection object from historyClass.Detecion class 

    Returns:
        DeepSortDetection: Detection object from deep_sort.deep_sort.detection.Detecion class 
    """
    return DeepSortDetection([(darknetDetection.X-darknetDetection.Width/2), 
        (darknetDetection.Y-darknetDetection.Height/2), 
        darknetDetection.Height, darknetDetection.Height], 
        float(darknetDetection.confidence), [], darknetDetection)

def updateHistory(history, DeepSortTracker, detections, historyDepth=30):
    """Update TrackedObject history

    Args:
        history (list[TrackedObject]): the history of tracked objects 
        DeepSortTracker (Tracker): deep_sort Tracker obj 
        detections (list[Detection]): list of new detections fresh from darknet 
        historyDepth (int) : number of detections stored in trackedObject.history 
    """
    wrapped_Detections = [makeDeepSortDetectionObject(det) for det in detections] 
    DeepSortTracker.predict()
    DeepSortTracker.update(wrapped_Detections)
    for track in DeepSortTracker.tracks:
        updated = False
        prevTO = None
        for trackedObject in history:
            if track.track_id == trackedObject.objID:
                if track.time_since_update == 0:
                    trackedObject.update(track.darknetDets[-1])
                    if len(trackedObject.history) > historyDepth:
                        trackedObject.history.remove(trackedObject.history[0])
                else:
                    # if arg in update is None, then time_since_update += 1
                    trackedObject.update()
                updated = True 
                prevTO = trackedObject
                break
        if prevTO is not None:
            if prevTO.max_age == prevTO.time_since_update:
                try:
                    history.remove(prevTO)
                    print(len(history))
                except:
                    print("Warning at removal of obj ID {}".format(prevTO.objID))
        if not updated:
            history.append(TrackedObject(track.track_id, track.darknetDets[-1], track._max_age))

# global var for adjusting stored history length
HISTORY_DEPTH = 30 
FUTUREPRED = 30

def main():
    args = parseArgs()
    input = args.input
    if input is not None:
        import hldnapi
    # check input source
    try:
        input = int(input)
    except ValueError:
        print("Input source is a Video.")
    # get video capture object
    cap = cv.VideoCapture(input)
    # exit if video cant be opened
    if not cap.isOpened():
        print("Source cannot be opened.")
        exit(0)
    # forward declaration of history(list[TrackedObject])
    history = []
    # generating colors for bounding boxes based on the class names of the neural net
    colors = class_colors(hldnapi.class_names)
    # get frame width
    frameWidth = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    # get frame height
    frameHeight = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    # create DeepSortTracker with command line arguments
    tracker = getDeepSortTracker(initDeepSortTrackerMetric(args.max_cosine_distance, args.nn_budget), historyDepth=HISTORY_DEPTH)
    # start main loop
    while(1):
        # get current frame from video
        ret, frame = cap.read()
        if frame is None:
            break
        # get current frame number
        frameNumber = cap.get(cv.CAP_PROP_POS_FRAMES)
        # time before computation
        prev_time = time.time()
        # use darknet neural net to detects objects
        detections = hldnapi.cvimg2detections(frame)
        # filter detections, only return the ones given in the targetNames tuple
        targets = getTargets(detections, frameNumber, targetNames=("person", "car"))
        # update track history
        updateHistory(history, tracker, targets)
        # draw bounding boxes of filtered detections
        draw_boxes(history, frame, colors, frameNumber)
        # run prediction algorithm and draw predictions on objects, that are in motion
        for obj in history:
            predictTraj(obj, linear_model.RANSACRegressor(), historyDepth=HISTORY_DEPTH, futureDepth=FUTUREPRED)
            draw_predictions(obj, frame, frameNumber)
        # show video frame
        cv.imshow("FRAME", frame)
        # calculating fps from time before computation and time now
        fps = int(1/(time.time() - prev_time))
        # print FPS to stdout
        print("FPS: {}".format(fps))
        # press 'q' to stop playing video
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
