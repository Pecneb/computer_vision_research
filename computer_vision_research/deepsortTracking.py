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
import sqlite3

import utility.databaseLoader as databaseLoader
import utility.databaseLogger as databaseLogger
from computer_vision_research.utility.dataManagementClasses import Detection as darknetDetection
from computer_vision_research.utility.dataManagementClasses import TrackedObject
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker


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

def getTracker(metricObj, max_iou_distance, historyDepth, db_connection = None):
    """DeepSort Tracker object fractory

    Args:
        metricObj (metric): DistanceMetric object for Tracker object from deep_sort.deep_sort.tracker.Tracker class 

    Returns:
        tracker: deep_sort Tracker object 
    """
    if db_connection is not None:
        return Tracker(metric=metricObj, max_age=10, historyDepth=historyDepth, _next_id=databaseLoader.queryLastObjID(db_connection), max_iou_distance=max_iou_distance)
    else:
        return Tracker(metric=metricObj, max_age=10, historyDepth=historyDepth, max_iou_distance=max_iou_distance)

def makeDetectionObject(darknetDetection: darknetDetection):
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

def updateHistory(trackedObjects: list, Tracker: Tracker, detections: list, db_connection = None, k_velocity=10, k_acceleration=2, historyDepth=30, joblibdb: list = None):
    """Update TrackedObject history

    Args:
        trackedObjects (list[TrackedObject]): the history of tracked objects 
        Tracker (Tracker): deep_sort Tracker obj 
        detections (list[Detection]): list of new detections fresh from darknet 
        db_connection (sqlite3.Connection): Connection object to database, to log objects.
        historyDepth (int) : number of detections stored in trackedObject.history 
    """
    wrapped_Detections = [makeDetectionObject(det) for det in detections] 
    Tracker.predict()
    Tracker.update(wrapped_Detections)
    for track in Tracker.tracks:
        updated = False
        for trackedObject in trackedObjects:
            if track.track_id == trackedObject.objID:
                if track.time_since_update == 0:
                    trackedObject.update(track.darknetDet, track.mean, k_velocity, k_acceleration)
                    if len(trackedObject.history) > historyDepth:
                        trackedObject.history.pop(0)
                else:
                    # if arg in update is None, then time_since_update += 1
                    trackedObject.update()
                    if trackedObject.max_age <= trackedObject.time_since_update:
                        if joblibdb is not None:
                            joblibdb.append(trackedObject)
                        trackedObjects.remove(trackedObject)
                updated = True 
                break
        if not updated:
            newTrack = TrackedObject(track.track_id, track.darknetDet, track._max_age)
            trackedObjects.append(newTrack)
            if db_connection is not None:
                databaseLogger.logObject(db_connection, newTrack.objID, newTrack.label)
            
