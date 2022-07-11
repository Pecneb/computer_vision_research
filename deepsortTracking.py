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
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort.detection import Detection
from historyClass import TrackedObject

def initTrackerMetric(max_cosine_distance, nn_budget, metric="cosine"):
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

def getTracker(metricObj, historyDepth):
    """DeepSort Tracker object fractory

    Args:
        metricObj (metric): DistanceMetric object for Tracker object from deep_sort.deep_sort.tracker.Tracker class 

    Returns:
        tracker: deep_sort Tracker object 
    """
    return Tracker(metricObj, historyDepth)

def makeDetectionObject(darknetDetection):
    """DeepSort Detection object factory

    Args:
        darknetDetection (Detection): Detection object from historyClass.Detecion class 

    Returns:
        Detection: Detection object from deep_sort.deep_sort.detection.Detecion class 
    """
    return Detection([(darknetDetection.X-darknetDetection.Width/2), 
        (darknetDetection.Y-darknetDetection.Height/2), 
        darknetDetection.Height, darknetDetection.Height], 
        float(darknetDetection.confidence), [], darknetDetection)

def updateHistory(history, Tracker, detections, historyDepth=30):
    """Update TrackedObject history

    Args:
        history (list[TrackedObject]): the history of tracked objects 
        Tracker (Tracker): deep_sort Tracker obj 
        detections (list[Detection]): list of new detections fresh from darknet 
        historyDepth (int) : number of detections stored in trackedObject.history 
    """
    wrapped_Detections = [makeDetectionObject(det) for det in detections] 
    Tracker.predict()
    Tracker.update(wrapped_Detections)
    for track in Tracker.tracks:
        updated = False
        prevTO = None
        for trackedObject in history:
            if track.track_id == trackedObject.objID:
                if track.time_since_update == 0:
                    trackedObject.update(track.darknetDets[-1], track.mean)
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
