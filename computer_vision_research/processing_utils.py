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
### System ###
import time
import os
import logging
from typing import List
from copy import deepcopy
from pathlib import Path

### Third Party ###
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import joblib 

### Local ###
import databaseLoader as databaseLoader
from classifier import OneVSRestClassifierExtended 
from visualizer import aoiextraction
import dataManagementClasses as dataManagementClasses

logging.basicConfig(filename="processing_utils.log", level=logging.DEBUG)

def savePlot(fig: plt.Figure, name: str):
    fig.savefig(name, dpi=150)

def detectionFactory(objID: int, frameNum: int, label: str, confidence: float, x: float, y: float, width: float, height: float, vx: float, vy: float, ax:float, ay: float):
    """Create Detection object.

    Args:
        objID (int): Object ID, to which object the Detection belongs to. 
        frameNum (int): Frame number when the Detection occured. 
        label (str): Label of the object, etc: car, person... 
        confidence (float): Confidence number, of how confident the neural network is in the detection. 
        x (float):  X coordinate of the object.
        y (float): Y coordinate of the object. 
        width (float): Width of the bounding box of the object. 
        height (float): Height of the bounging box of the object. 
        vx (float): Velocity on the X axis. 
        vy (float): Velocity on the Y axis. 
        ax (float): Acceleration on the X axis. 
        ay (float): Acceleration on the Y axis. 

    Returns:
        Detection: The Detection object, which is to be returned. 
    """
    retDet = dataManagementClasses.Detection(label, confidence, x,y,width,height,frameNum)
    retDet.objID = objID
    retDet.VX = vx
    retDet.VY = vy
    retDet.AX = ax
    retDet.AY = ay
    return retDet

def trackedObjectFactory(detections: tuple):
    """Create trackedObject object from list of detections

    Args:
        detections (list): list of detection 

    Returns:
        TrackedObject:  trackedObject
    """
    history, history_X, history_Y, history_VX_calculated, history_VY_calculated, history_AX_calculated, history_AY_calculated = detections
    tmpObj = dataManagementClasses.TrackedObject(history[0].objID, history[0], len(detections))
    tmpObj.label = detections[0][-1].label
    tmpObj.history = history
    tmpObj.history_X = history_X
    tmpObj.history_Y = history_Y
    tmpObj.history_VX_calculated = history_VX_calculated
    tmpObj.history_VY_calculated = history_VY_calculated
    tmpObj.history_AX_calculated = history_AX_calculated
    tmpObj.history_AY_calculated = history_AY_calculated
    tmpObj.X = detections[0][-1].X
    tmpObj.Y = detections[0][-1].Y
    tmpObj.VX = detections[0][-1].VX
    tmpObj.VY = detections[0][-1].VY
    tmpObj.AX = detections[0][-1].AX
    tmpObj.AY = detections[0][-1].AY
    return tmpObj

def cvCoord2npCoord(Y: np.ndarray) -> np.ndarray:
    """Convert OpenCV Y axis coordinates to numpy coordinates.

    Args:
        Y (np.ndarray): Y axis coordinate vector

    Returns:
        np.ndarray: Y axis coordinate vector
    """
    return 1 - Y

def detectionParser(rawDetectionData) -> tuple:
    """Convert raw detection data loaded from db to class Detection and numpy arrays.

    Args:
        rawDetectionData (list): Raw values loaded from db 

    Returns:
        tuple: tuple containing detections, and all the history numpy arrays  
    """
    detections = []
    history_X = np.array([])
    history_Y = np.array([])
    history_VX_calculated = np.array([])
    history_VY_calculated = np.array([])
    history_AX_calculated = np.array([])
    history_AY_calculated = np.array([])
    for entry in rawDetectionData:
        detections.append(detectionFactory(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7], entry[8], entry[9], entry[10], entry[11]))
        history_X = np.append(history_X, [entry[3]])
        history_Y = np.append(history_Y, [entry[4]])
        history_VX_calculated = np.append(history_VX_calculated, [entry[12]])
        history_VY_calculated = np.append(history_VY_calculated, [entry[13]])
        history_AX_calculated = np.append(history_AX_calculated, [entry[14]])
        history_AY_calculated = np.append(history_AY_calculated, [entry[15]])
    return (detections, history_X, history_Y, history_VX_calculated, history_VY_calculated, history_AX_calculated, history_AY_calculated)

def parseRawObject2TrackedObject(rawObjID: int, path2db: str):
    """Takes an objID and the path 2 database, then returns a trackedObject object if detections can be assigned to the object.

    Args:
        rawObjID (int): ID of an object 
        path2db (str): path to database 

    Returns:
        trackedObject: trackedObject object from dataManagement class, if no dets can be assigned to it, then returns False 
    """
    rawDets = databaseLoader.loadDetectionsOfObject(path2db, rawObjID)
    if len(rawDets) > 0:
        logging.debug(f"Detections loaded: {len(rawDets)} {rawDets[0]}")
        retTO = trackedObjectFactory(detectionParser(rawDets))
        return retTO
    else:
        return False

def preprocess_database_data(path2db: str):
    """Preprocessing database data (detections). Assigning detections to objects.

    Args:
        path2db (str): Path to database file. 

    Returns:
        list: list of object tracks 
    """
    rawObjectData = databaseLoader.loadObjects(path2db)
    trackedObjects = []
    for rawObj in tqdm.tqdm(rawObjectData, desc="Loading detections of tracks."):
        tmpDets = []
        rawDets = databaseLoader.loadDetectionsOfObject(path2db, rawObj[0])
        if len(rawDets) > 0:
            tmpDets = detectionParser(rawDets)
            trackedObjects.append(trackedObjectFactory(tmpDets))
    return trackedObjects

def preprocess_database_data_multiprocessed(path2db: str, n_jobs=None):
    """Preprocessing database data (detections). Assigning detections to objects.
    This is the multoprocessed variant of the preprocess_database_data() func.

    Args:
        path2db (str): Path to database file. 

    Returns:
        list: list of object tracks 
    """
    from multiprocessing import Pool
    rawObjectData = databaseLoader.loadObjects(path2db)
    tracks = []
    with Pool(processes=n_jobs) as pool:
        print("Preprocessing started.")
        start = time.time()
        results = pool.starmap_async(parseRawObject2TrackedObject, [[rawObj[0], path2db] for rawObj in rawObjectData])
        for result in tqdm.tqdm(results.get(), desc="Unpacking the result of detection assignment."):
            if result:
                tracks.append(result)
                logging.debug(f"{len(tracks)}")
        print(f"Detections assigned to Objects in {time.time()-start}s")
    return tracks
 
def makeColormap(path2db):
    """Make colormap based on number of objects logged in the database.
    This colormap vector will be input to matplotlib scatter plot.

    Args:
        path2db (str): Path to database 

    Returns:
        colormap: list of color gradient vectors (R,G,B)
    """
    objects = databaseLoader.loadObjects(path2db)
    detections = databaseLoader.loadDetections(path2db)
    objectVector = []
    detectionVector = []
    [objectVector.append(obj) for obj in objects]
    [detectionVector.append(det) for det in detections]
    colors = np.random.rand(len(objectVector))
    colormap = np.zeros(shape=(len(detectionVector)))
    for i in range(len(objectVector)):
        for j in range(len(detectionVector)):
            if objectVector[i][0] == detectionVector[j][0]:
                colormap[j] = colors[i]
    return colormap

def findEnterAndExitPoints(path2db: str):
    """Extracting only the first and the last detections of tracked objects.

    Args:
        path2db (str): Path to the database file. 

    Returns:
        enterDetection, exitDetections: List of first and last detections of objects. 
    """
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    rawObjectData = databaseLoader.loadObjects(path2db)
    trackedObjects = []
    for obj in tqdm.tqdm(rawObjectData, desc="Filter out enter and exit points."):
        tmpDets = []
        for det in detections:
           if det.objID == obj[0]:
            tmpDets.append(det)
        if len(tmpDets) > 0:
            trackedObjects.append(trackedObjectFactory(tmpDets))
    enterDetections = [obj.history[0] for obj in trackedObjects]
    exitDetections = [obj.history[-1] for obj in trackedObjects]
    return enterDetections, exitDetections 

def order2track(args: list):
    """Orders enter and exit dets to object.

    Args:
        args (list): list of args, that should look like [detections, objID] 

    Returns:
        tuple: tuple of dets, enter det and exit det 
        bool: returns False, when tuple is empty
    """
    dets = []
    for det in args[0]:
        if det.objID == args[1]:
            dets.append(det)
    if dets:
        return (dets[0], dets[-1])
    else:
        return False

def findEnterAndExitPointsMultiprocessed(path2db: str, n_jobs=None):
    """Extract only the first and last detections of tracked objects.
    This is a multithreaded implementation of function findEnterAndExitPoints(path2db: str)

    Args:
        path2db (str): Path to database file. 
    """
    from multiprocessing import Pool
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    rawObjectData = databaseLoader.loadObjects(path2db)
    enterDetections = []
    exitDetections = []
    iterable = [[detections, obj[0]] for obj in rawObjectData]
    with Pool(processes=n_jobs) as executor:
        start = time.time()
        results = executor.map(order2track, iterable) 
        print("Enter and Exit points with multiprocessing: %f"%(time.time()-start))
        for result in tqdm.tqdm(results, desc="Enter and Exit points."):
            if result:
                enterDetections.append(result[0])
                exitDetections.append(result[1])
    return enterDetections, exitDetections

def euclidean_distance(q1: float, p1: float, q2: float, p2: float):
    """Calculate euclidean distance of 2D vectors.

    Args:
        q1 (float): First element of vector1. 
        p1 (float): First element of vector2. 
        q2 (float): Second element of vector1. 
        p2 (float): Second element of vector2.
        
    Returns:
        float: euclidean distance of the two vectors. 
    """
    return (((q1-p1)**2)+((q2-p2)**2))**0.5
    

def filter_out_false_positive_detections_by_enter_exit_distance(trackedObjects: list, threshold: float):
    """This function is to filter out false positive detections, 
    that enter and exit points are closer, than the threshold value given in the arguments.

    Args:
        trackedObjects (list): list of object trackings 
        threshold (float): if enter and exit points are closer than this value, they are exclueded from the return list

    Returns:
        list: list of returned filtered tracks 
    """
    filteredTracks = []
    for obj in tqdm.tqdm(trackedObjects, desc="False positive filtering."):
        d = euclidean_distance(obj.history[0].X, obj.history[-1].X, obj.history[0].Y, obj.history[-1].Y)
        if d > threshold:
            filteredTracks.append(obj)
    return filteredTracks

def is_noise(trackedObject, threshold: float = 0.1):
    for i in range(len(trackedObject.history_X)-1):
        d = np.linalg.norm(np.array([trackedObject.history_X[i], trackedObject.history_Y[i]])-np.array([trackedObject.history_X[i+1], trackedObject.history_Y[i+1]]))
        if d > threshold:
            return True
    return False
    
def filter_out_noise_trajectories(trackedObjects: list, distance: float = 0.05):
    filtered = []
    for i in range(len(trackedObjects)):
        if not is_noise(trackedObjects[i], distance):
            filtered.append(trackedObjects[i])
    return np.array(filtered)

def search_min_max_coordinates(trackedObjects: list):
    """Search for minimum and maximum of x,y coordinates,
    to find the edges of the range of interest.

    Args:
        trackedObjects (list): list of trajectory objects 

    Returns:
        tuple: tuple consisting of min x,y and max x,y coordinates 
    """
    max_y = 0 
    min_y = 9999 
    max_x = 0
    min_x = 9999
    coord = np.array([]).reshape((0,2))
    X = np.array([])
    Y = np.array([])
    for obj in tqdm.tqdm(trackedObjects, desc="Looking for min max values."):
        X = np.append(X, obj.history_X)
        Y = np.append(Y, obj.history_Y)
    min_x = np.min(X)
    max_x = np.max(X)
    min_y = np.min(Y)
    max_y = np.max(Y)
    return min_x, min_y, max_x, max_y

def filter_out_edge_detections(trackedObjects: list, threshold: float):
    """Filter out objects, that enter and exit detections coordinates are in the threshold value.

    Args:
        trackedObjects (list): list of object trackings 
        threshold (float): objects only under this value will be returned 

    Returns:
        list: filtered list of tracks 
    """
    min_x, min_y, max_x, max_y = search_min_max_coordinates(trackedObjects)
    #print(f"\n Max X: {max_x}\n Min X: {min_x}\n Max Y: {max_y}\n Min Y: {min_y}\n")
    #print(f"\n Thresholds:\n Max X: {max_x-threshold}\n Min X: {min_x+threshold}\n Max Y: {max_y-threshold}\n Min Y: {min_y+threshold}\n")
    filteredTracks = []
    for obj in tqdm.tqdm(trackedObjects, desc="Filter out edge detections."):
        if (((obj.history[0].X <= min_x+threshold or obj.history[0].X >= max_x-threshold) or 
            (obj.history[0].Y <= min_y+threshold or obj.history[0].Y >= max_y-threshold)) and
            ((obj.history[-1].X <= min_x+threshold or obj.history[-1].X >= max_x-threshold) or 
            (obj.history[-1].Y <= min_y+threshold or obj.history[-1].Y >= max_y-threshold))): 
            filteredTracks.append(obj)
    # even though we did the edge filtering, we can run an euclidean distance based filtering, which's threshold is hardcoded for now
    return filteredTracks

def filter_trajectories(trackedObjects: list, threshold: float = 0.7, enter_exit_dist: float = 0.4, detectionDistanceFiltering: bool = True, detDist: float = 0.05):
    """Run filtering process on trajectory datase.

    Args:
        trackedObjects (list): Trajectory dataset 
        threshold (float): threshold for min max search

    Returns:
        denoised_trajectories: filtered trajectories without noice 
    """
    edge_trajectories = filter_out_edge_detections(trackedObjects, threshold)
    filtered_trajectories = filter_out_false_positive_detections_by_enter_exit_distance(edge_trajectories, enter_exit_dist)
    if detectionDistanceFiltering:
        denoised_trajectories = filter_out_noise_trajectories(filtered_trajectories, detDist)
        return np.array(denoised_trajectories, dtype=object)
    return np.array(filtered_trajectories, dtype=object)

def save_filtered_dataset(dataset: str | Path, threshold: float, max_dist: float, euclidean_filtering: bool = False, outdir = None):
    """Filter dataset and save to joblib binary.

    Args:
        dataset (str): Dataset path 
        threshold (float): Filter threshold 
    """
    if "filtered" in str(dataset):
        print(f"{dataset} is already filtered.")
        return False
    try:
        trajectories = load_dataset(dataset)
    except EOFError as e:
        print(e)
        return False
    logging.info(f"Trajectories \"{dataset}\" loaded")
    trajs2save = filter_trajectories(
        trackedObjects=trajectories, 
        threshold=threshold, 
        detectionDistanceFiltering=euclidean_filtering, 
        detDist=max_dist
    )
    logging.info(f"Dataset filtered. Trajectories reduced from {len(trajectories)} to {len(trajs2save)}")
    datasetPath = Path(dataset)
    if outdir is not None:
        outPath = Path(outdir)
        if not outPath.exists():
            outPath.mkdir()
        filteredDatasetPath = outPath.joinpath(datasetPath.stem+"_filtered.joblib")
    else:
        filteredDatasetPath = datasetPath.parent.joinpath(datasetPath.stem+"_filtered.joblib")
    joblib.dump(trajs2save, filteredDatasetPath)
    logging.info(f"Filtered dataset saved to {filteredDatasetPath.absolute()}")

def filter_by_class(trackedObjects: list, label="car"):
    """Only return objects with given label.

    Args:
        trackedObjects (list): list of tracked objects 
        label (str, optional): label of object. Defaults to "car".

    Returns:
        list: list of "label" objects 
    """
    return [obj for obj in trackedObjects if obj.label==label]


def makeFeatureVectors_Nx2(trackedObjects: list) -> np.ndarray:
    """Create 2D feature vectors from tracks.
    The enter and exit coordinates are put in different vectors. Only creating 2D vectors.

    Args:
        trackedObjects (list): list of tracked objects 

    Returns:
        np.ndarray: numpy array of feature vectors 
    """
    featureVectors = [] 
    for obj in trackedObjects:
        featureVectors.append(obj.history[0].X, obj.history[0].Y)
        featureVectors.append(obj.history[-1].X, obj.history[-1].Y)
    return np.array(featureVectors)

def make_2D_feature_vectors(trackedObjects: list) -> np.ndarray:
    """Create 2D feature vectors from tracks.
    The enter and exit coordinates are put in one vector. Creating 4D vectors.
    v = [exitX, exitY]

    Args:
        trackedObjects (list): list of tracked objects 

    Returns:
        np.ndarray: numpy array of feature vectors 
    """
    featureVectors = np.array([np.array([obj.history[-1].X, obj.history[-1].Y]) for obj in tqdm.tqdm(trackedObjects, desc="Feature vectors.")])
    return featureVectors

def make_4D_feature_vectors(trackedObjects: list) -> np.ndarray:
    """Create 4D feature vectors from tracks.
    The enter and exit coordinates are put in one vector. Creating 4D vectors.
    v = [enterX, enterY, exitX, exitY]

    Args:
        trackedObjects (list): list of tracked objects 

    Returns:
        np.ndarray: numpy array of feature vectors 
    """
    featureVectors = np.array([np.array([obj.history[0].X, obj.history[0].Y, obj.history[-1].X, obj.history[-1].Y]) for obj in tqdm.tqdm(trackedObjects, desc="Feature vectors.")])
    return featureVectors

def make_6D_feature_vectors(trackedObjects: list) -> np.ndarray:
    """Create 6D feature vectors from tracks.
    The enter, middle and exit coordinates are put in one vector. Creating 6D vectors.
    v = [enterX, enterY, exitX, exitY]

    Args:
        trackedObjects (list): list of tracked objects 

    Returns:
        np.ndarray: numpy array of feature vectors 
    """
    featureVectors = np.array([np.array([obj.history[0].X, obj.history[0].Y, obj.history[len(obj.history) // 2].X, obj.history[len(obj.history) // 2].Y, obj.history[-1].X, obj.history[-1].Y]) for obj in tqdm.tqdm(trackedObjects, desc="Feature vectors.")])
    return featureVectors

def shuffle_data(trackedObjects: list) -> list:
    rng = np.random.default_rng()
    for i in range(len(trackedObjects)):
        randIDX = int(len(trackedObjects) * rng.random())
        tmpObj = trackedObjects[i]
        trackedObjects[i] = trackedObjects[randIDX]
        trackedObjects[randIDX] = tmpObj
    
def test_shuffle(path2db: str):
    origObjects = preprocess_database_data_multiprocessed(path2db=path2db, n_jobs=16)
    objectsToShuffle = origObjects.copy()
    shuffle_data(objectsToShuffle)
    changed = 0
    for i in range(len(origObjects)):
        if origObjects[i].objID != objectsToShuffle[i].objID:
            changed += 1
    print(f"Entropy = {changed/len(origObjects)}")


def checkDir(path2db):
    """Check for dir of given database, to be able to save plots.

    Args:
        path2db (str): Path to database. 
    """
    if not os.path.isdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0])):
        os.mkdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0]))
        print("Directory \"research_data/{}\" is created.".format(path2db.split('/')[-1].split('.')[0]))

def make_features_for_classification(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors for classification algorithm

    Args:
        trackedObjects (list): Tracked objects 
        k (int): K is the number of slices, the object history should be sliced up into.
        labels (np.ndarray): Results of clustering.

    Returns:
        np.ndarray: featurevectors and the new labels to fit the featurevectors 
    """
    featureVectors = []
    newLabels = []
    for j in range(len(trackedObjects)):
        step = len(trackedObjects[j].history)//k
        if step > 0:
            midstep = step//2
            for i in range(0, len(trackedObjects[j].history)-step, step):
                featureVectors.append(np.array([trackedObjects[j].history[i].X,trackedObjects[j].history[i].Y,trackedObjects[j].history[i+midstep].X,trackedObjects[j].history[i+midstep].Y,trackedObjects[j].history[i+step].X,trackedObjects[j].history[i+step].Y]))
                newLabels.append(labels[j])
    return np.array(featureVectors), np.array(newLabels)

def make_features_for_classification_velocity(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors for classification algorithm

    Args:
        trackedObjects (list): Tracked objects 
        k (int): K is the number of slices, the object history should be sliced up into.
        labels (np.ndarray): Results of clustering.

    Returns:
        np.ndarray, np.ndarray: featureVectors, labels 
    """
    featureVectors = []
    newLabels = []
    for j in range(len(trackedObjects)):
        step = len(trackedObjects[j].history)//k
        if step > 0:
            midstep = step//2
            for i in range(0, len(trackedObjects[j].history)-step, step):
                featureVectors.append(np.array([trackedObjects[j].history[i].X,trackedObjects[j].history[i].Y,trackedObjects[j].history[i].VX,trackedObjects[j].history[i].VY,trackedObjects[j].history[i+midstep].X,trackedObjects[j].history[i+midstep].Y,trackedObjects[j].history[i+step].X,trackedObjects[j].history[i+step].Y,trackedObjects[j].history[i+step].VX,trackedObjects[j].history[i+step].VY]))
                newLabels.append(labels[j])
    return np.array(featureVectors), np.array(newLabels)

def make_feature_vectors_version_one(trackedObjects: list, k: int, labels: np.ndarray, reduced_labels: np.ndarray = None, up_until: float = 1):
    """Make feature vectors for classification algorithm

    Args:
        trackedObjects (list): Tracked objects 
        k (int): K is the number of slices, the object history should be sliced up into.
        labels (np.ndarray): Results of clustering.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: featureVectors, labels, timeOfFeatureVectors
    """
    featureVectors = []
    newLabels = []
    newReducedLabels = []
    track_history_metadata = [] # list of [start_time, mid_time, end_time, history_length, trackID]
    #TODO remove time vector, use track_history_metadata instead
    for j in range(len(trackedObjects)):
        step = len(trackedObjects[j].history)//k
        if step > 0:
            midstep = step//2
            for i in range(0, int(len(trackedObjects[j].history)*up_until)-step, step):
                featureVectors.append(np.array([trackedObjects[j].history[i].X, trackedObjects[j].history[i].Y, 
                                            trackedObjects[j].history[i].VX, trackedObjects[j].history[i].VY,
                                            trackedObjects[j].history[i+midstep].X, trackedObjects[j].history[i+midstep].Y,
                                            trackedObjects[j].history[i+step].X, trackedObjects[j].history[i+step].Y,
                                            trackedObjects[j].history[i+step].VX, trackedObjects[j].history[i+step].VY]))
                newLabels.append(labels[j])
                newReducedLabels.append(reduced_labels[j])
                track_history_metadata.append([trackedObjects[j].history[i].frameID, trackedObjects[j].history[i+midstep].frameID, 
                trackedObjects[j].history[i+step].frameID, len(trackedObjects[j].history), trackedObjects[j].objID])
    return np.array(featureVectors), np.array(newLabels), np.array(track_history_metadata), np.array(newReducedLabels)

def make_feature_vectors_version_one_half(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors for classification algorithm

    Args:
        trackedObjects (list): Tracked objects 
        k (int): K is the number of slices, the object history should be sliced up into.
        labels (np.ndarray): Results of clustering.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: featureVectors, labels, timeOfFeatureVectors
    """
    featureVectors = []
    newLabels = []
    track_history_metadata = [] # list of [start_time, mid_time, end_time, history_length, trackID]
    #TODO remove time vector, use track_history_metadata instead
    for j in tqdm.tqdm(range(len(trackedObjects)), desc="Features for classification."):
        step = (len(trackedObjects[j].history)//2)//k
        if step > 0:
            midstep = step//2
            for i in range(len(trackedObjects[j].history)//2, len(trackedObjects[j].history)-step, step):
                featureVectors.append(np.array([trackedObjects[j].history[i].X,trackedObjects[j].history[i].Y,
                                                trackedObjects[j].history[i].VX,trackedObjects[j].history[i].VY,
                                                trackedObjects[j].history[i+midstep].X,trackedObjects[j].history[i+midstep].Y,
                                                trackedObjects[j].history[i+step].X,trackedObjects[j].history[i+step].Y,
                                                trackedObjects[j].history[i+step].VX,trackedObjects[j].history[i+step].VY]))
                newLabels.append(labels[j])
                track_history_metadata.append([trackedObjects[j].history[i].frameID, trackedObjects[j].history[i+midstep].frameID, 
                                                trackedObjects[j].history[i+step].frameID, len(trackedObjects[j].history), trackedObjects[j].objID])
    return np.array(featureVectors), np.array(newLabels), np.array(track_history_metadata)

def make_feature_vectors_version_two(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors from track histories, such as starting from the first detection incrementing the vectors length by a given factor, building multiple vectors from one history.
    A vector is made up from the absolute first detection of the history, a relative middle detection, and a last detecion, that's index is incremented, for the next feature vector until 
    this last detection reaches the end of the history. Next to the coordinates, also the velocity of the object is being included in the feature vector.

    Args:
        trackedObjects (list): Tracked objects. 
        labels (np.ndarray): Labels of the tracks, which belongs to a given cluster, given by the clustering algo. 

    Returns:
        tuple of numpy arrays: The newly created feature vectors, the labels created for each feature vector, and the metadata that contains the information of time frames, and to which object does the feature belongs to. 
    """
    X_featurevectors = [] # [history[0].X, history[0]. Y,history[0].VX, history[0].VY,history[mid].X, history[mid].Y,history[end].X, history[end]. Y,history[end].VX, history[end].VY]
    y_newLabels = []
    featurevector_metadata = [] # [start_time, mid_time, end_time, history_length, trackID]
    for i, track in tqdm.tqdm(enumerate(trackedObjects), desc="Features for classification.", total=len(trackedObjects)):
        step = (len(track.history))//k
        if step >= 2:
            for j in range(step, len(track.history), step):
                midx = j//2
                X_featurevectors.append(np.array([track.history[0].X, track.history[0].Y, 
                                                track.history[0].VX, track.history[0].VY, 
                                                track.history[midx].X, track.history[midx].Y, 
                                                track.history[j].X, track.history[j].Y, 
                                                track.history[j].VX, track.history[j].VY])) 
                y_newLabels.append(labels[i])
                featurevector_metadata.append(np.array([track.history[0].frameID, track.history[midx].frameID, 
                                            track.history[j].frameID, len(track.history), track.objID]))
    return np.array(X_featurevectors), np.array(y_newLabels), np.array(featurevector_metadata)

def make_feature_vectors_version_two_half(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors from track histories, such as starting from the first detection incrementing the vectors length by a given factor, building multiple vectors from one history.
    A vector is made up from the absolute first detection of the history, a relative middle detection, and a last detecion, that's index is incremented, for the next feature vector until 
    this last detection reaches the end of the history. Next to the coordinates, also the velocity of the object is being included in the feature vector.

    Args:
        trackedObjects (list): Tracked objects. 
        labels (np.ndarray): Labels of the tracks, which belongs to a given cluster, given by the clustering algo. 

    Returns:
        tuple of numpy arrays: The newly created feature vectors, the labels created for each feature vector, and the metadata that contains the information of time frames, and to which object does the feature belongs to. 
    """
    X_featurevectors = []
    y_newLabels = []
    featurevector_metadata = [] # [start_time, mid_time, end_time, history_length, trackID]
    for i, track in tqdm.tqdm(enumerate(trackedObjects), desc="Features for classification.", total=len(trackedObjects)):
        step = (len(trackedObjects[i].history))//k
        if step >= 2:
            for j in range((len(trackedObjects[i].history)//2)+step, len(trackedObjects[i].history), step):
                midx = j//2
                X_featurevectors.append(np.array([trackedObjects[i].history[0].X, trackedObjects[i].history[0].Y, 
                                                trackedObjects[i].history[0].VX, trackedObjects[i].history[0].VY, 
                                                trackedObjects[i].history[midx].X, trackedObjects[i].history[midx].Y, 
                                                trackedObjects[i].history[j].X, trackedObjects[i].history[j].Y, 
                                                trackedObjects[i].history[j].VX, trackedObjects[i].history[j].VY])) 
                y_newLabels.append(labels[i])
                featurevector_metadata.append(np.array([trackedObjects[i].history[0].frameID, trackedObjects[i].history[midx].frameID, 
                                            trackedObjects[i].history[j].frameID, len(trackedObjects[i].history), trackedObjects[i].objID]))
    return np.array(X_featurevectors), np.array(y_newLabels), np.array(featurevector_metadata)

def make_feature_vectors_version_three(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors from track histories, such as starting from the first detection incrementing the vectors length by a given factor, building multiple vectors from one history.
    A vector is made up from the absolute first detection of the history, a relative middle detection, and a last detecion, that's index is incremented, for the next feature vector until 
    this last detection reaches the end of the history.

    Args:
        trackedObjects (list): Tracked objects. 
        labels (np.ndarray): Labels of the tracks, which belongs to a given cluster, given by the clustering algo. 

    Returns:
        tuple of numpy arrays: The newly created feature vectors, the labels created for each feature vector, and the metadata that contains the information of time frames, and to which object does the feature belongs to. 
    """
    X_featurevectors = []
    y_newLabels = []
    featurevector_metadata = [] # [start_time, mid_time, end_time, history_length, trackID]
    for i in tqdm.tqdm(range(len(trackedObjects)), desc="Features for classification."):
        step = (len(trackedObjects[i].history))//k
        if step >= 2:
            for j in range(step, len(trackedObjects[i].history), step):
                midx = j//2
                fv = np.array([
                            trackedObjects[i].history[0].X, trackedObjects[i].history[0].Y, 
                            trackedObjects[i].history[midx].X, trackedObjects[i].history[midx].Y, 
                            trackedObjects[i].history[j].X, trackedObjects[i].history[j].Y])
                X_featurevectors.append(fv)
                y_newLabels.append(labels[i])
                featurevector_metadata.append(np.array([trackedObjects[i].history[0].frameID, trackedObjects[i].history[midx].frameID, 
                                            trackedObjects[i].history[j].frameID, len(trackedObjects[i].history), trackedObjects[i].objID]))
    return np.array(X_featurevectors), np.array(y_newLabels), np.array(featurevector_metadata)

def make_feature_vectors_version_three_half(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors from track histories, such as starting from the first detection incrementing the vectors length by a given factor, building multiple vectors from one history.
    A vector is made up from the absolute first detection of the history, a relative middle detection, and a last detecion, that's index is incremented, for the next feature vector until 
    this last detection reaches the end of the history. 

    Args:
        trackedObjects (list): Tracked objects. 
        labels (np.ndarray): Labels of the tracks, which belongs to a given cluster, given by the clustering algo. 

    Returns:
        tuple of numpy arrays: The newly created feature vectors, the labels created for each feature vector, and the metadata that contains the information of time frames, and to which object does the feature belongs to. 
    """
    X_featurevectors = []
    y_newLabels = []
    featurevector_metadata = [] # [start_time, mid_time, end_time, history_length, trackID]
    for i in tqdm.tqdm(range(len(trackedObjects)), desc="Features for classification."):
        step = (len(trackedObjects[i].history))//k
        if step >= 2:
            for j in range((len(trackedObjects[i].history)//2)+step, len(trackedObjects[i].history), step):
                midx = j//2
                X_featurevectors.append(np.array([trackedObjects[i].history[0].X, trackedObjects[i].history[0].Y, 
                                                trackedObjects[i].history[midx].X, trackedObjects[i].history[midx].Y, 
                                                trackedObjects[i].history[j].X, trackedObjects[i].history[j].Y]))
                y_newLabels.append(labels[i])
                featurevector_metadata.append(np.array([trackedObjects[i].history[0].frameID, trackedObjects[i].history[midx].frameID, 
                                            trackedObjects[i].history[j].frameID, len(trackedObjects[i].history), trackedObjects[i].objID]))
    return np.array(X_featurevectors), np.array(y_newLabels), np.array(featurevector_metadata)

def make_feature_vectors_version_four(trackedObjects: list, max_stride: int, labels: np.ndarray):
    """Make multiple feature vectors from one object's history. When max_stride is reached, use sliding window method to create the vectors.

    Args:
        trackedObjects (list): list of tracked objects 
        max_stride (int): max window size 
        labels (np.ndarray): cluster label of each tracked object 

    Returns:
        _type_: _description_
    """
    X_feature_vectors = np.array([])
    y_new_labels = np.array([])
    metadata = []
    for i, t in tqdm.tqdm(enumerate(trackedObjects), desc="Features for classification.", total=len(trackedObjects)):
        stride = 3
        if stride > t.history_X.shape[0]:
            continue
        for j in range(t.history_X.shape[0]-max_stride):
            if stride < max_stride:
                midx = stride // 2 
                end_idx = stride-1
                X_feature_vectors = np.append(X_feature_vectors, np.array([
                    t.history_X[0], t.history_Y[0], # enter coordinates
                    t.history_X[midx], t.history_Y[midx], # mid 
                    t.history_X[end_idx], t.history_Y[end_idx] # exit
                ])).reshape(-1, 6)
                metadata.append(np.array([t.history[0].frameID, t.history[midx].frameID, 
                                            t.history[end_idx].frameID, t.history_X.shape[0], t.objID]))
                stride += 1
            else:
                midx = j + (stride // 2)
                end_idx = j + stride-1
                X_feature_vectors = np.append(X_feature_vectors, np.array([
                    t.history_X[j], t.history_Y[j], # enter coordinates
                    t.history_X[midx], t.history_Y[midx], # mid 
                    t.history_X[end_idx], t.history_Y[end_idx] # exit
                ])).reshape(-1, 6)
                metadata.append(np.array([t.history[j].frameID, t.history[midx].frameID, 
                                            t.history[end_idx].frameID, t.history_X.shape[0], t.objID]))
            y_new_labels = np.append(y_new_labels, labels[i])
    return np.array(X_feature_vectors), np.array(y_new_labels), np.array(metadata)

def insert_weights_into_feature_vector(start: int, stop: int, n_weights: int, X: np.ndarray, Y: np.ndarray, insert_idx: int, feature_vector: np.ndarray):
    """Insert coordinates into feature vector starting from the start_insert_idx index.

    Args:
        start (int): first index of inserted coordinates 
        stop (int): stop index of coordinate vectors, which will not be inserted, this is the open end of the limits
        n_weights (int): number of weights to be inserted
        X (ndarray): x coordinate array
        Y (ndarray): y coordinate array
        start_insert_idx (int): the index where the coordinates will be inserted into the feature vector 
    """
    retv = feature_vector.copy()
    stepsize = (stop-start)//n_weights
    assert n_weights, f"n_weights={n_weights} and max_stride are not compatible, lower n_weights or increase max_stride"
    weights_inserted = 0
    for widx in range(stop-1, start-1, -stepsize):
        if weights_inserted == n_weights:
            break
        retv = np.insert(retv, insert_idx, [X[widx], Y[widx]])
        weights_inserted += 1
    return retv

def make_feature_vectors_version_five(trackedObjects: list, labels: np.ndarray, max_stride: int, n_weights: int):
    X_feature_vectors = np.array([])
    y_new_labels = np.array([])
    metadata = []
    for i, t in tqdm.tqdm(enumerate(trackedObjects), desc="Features for classification.", total=len(trackedObjects)):
        stride = max_stride
        if stride > t.history_X.shape[0]:
            continue
        for j in range(0, t.history_X.shape[0]-max_stride):
            """
            if stride < max_stride:
                midx = stride // 2
                end_idx = stride-1
                feature_vector = np.array([t.history_X[0], t.history_Y[0],
                                        t.history_X[midx], t.history_Y[midx],
                                        t.history_X[end_idx], t.history_Y[end_idx]])
                #feature_vector = insert_weights_into_feature_vector(midx, end_idx, n_weights, t.history_X, t.history_Y, 2, feature_vector)
                metadata.append(np.array([t.history[0].frameID, t.history[midx].frameID, 
                                            t.history[end_idx].frameID, t.history_X.shape[0], t.objID]))
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array([np.array([feature_vector])])
                else:
                    X_feature_vectors = np.append(X_feature_vectors, np.array([[feature_vector]]), axis=0)
                stride += 1
            else:
            """
            midx = j + (stride // 2) - 1
            end_idx = j + stride - 1
            feature_vector = np.array([t.history_X[j], t.history_Y[j],
                                    t.history_X[end_idx], t.history_Y[end_idx]])
            feature_vector = insert_weights_into_feature_vector(midx, end_idx, n_weights, t.history_X, t.history_Y, 2, feature_vector)
            feature_vector = insert_weights_into_feature_vector(midx, end_idx, n_weights, t.history_VX_calculated, t.history_VY_calculated, 2, feature_vector)
            if X_feature_vectors.shape == (0,):
                X_feature_vectors = np.array(feature_vector).reshape((-1,4+(n_weights*4)))
            else:
                X_feature_vectors = np.append(X_feature_vectors, np.array([feature_vector]), axis=0)
            metadata.append(np.array([t.history[j].frameID, t.history[midx].frameID, 
                                        t.history[end_idx].frameID, t.history_X.shape[0], t.objID]))
            y_new_labels = np.append(y_new_labels, labels[i])
    return np.array(X_feature_vectors), np.array(y_new_labels, dtype=int), np.array(metadata)

def make_feature_vectors_version_six(trackedObjects: list, labels: np.ndarray, max_stride: int, weights: np.ndarray):
    if weights.shape != (12,):
        raise ValueError("Shape of weights must be equal to shape(12,).")
    X_feature_vectors = np.array([])
    y_new_labels = np.array([])
    metadata = []
    for i, t in tqdm.tqdm(enumerate(trackedObjects), desc="Features for classification.", total=len(trackedObjects)):
        stride = max_stride
        if stride > t.history_X.shape[0]:
            continue
        for j in range(0, t.history_X.shape[0]-max_stride, max_stride):
            midx = j + (3*stride // 4) - 1
            end_idx = j + stride - 1
            feature_vector = np.array([t.history_X[j], t.history_Y[j], t.history_VX_calculated[j], t.history_VY_calculated[j],
                                    t.history_X[midx], t.history_Y[midx], t.history_VX_calculated[midx], t.history_VY_calculated[midx],
                                    t.history_X[end_idx], t.history_Y[end_idx], t.history_VX_calculated[end_idx], t.history_VY_calculated[end_idx]]) * weights
            if X_feature_vectors.shape == (0,):
                X_feature_vectors = np.array(feature_vector).reshape((-1,12))
            else:
                X_feature_vectors = np.append(X_feature_vectors, np.array([feature_vector]), axis=0)
            metadata.append(np.array([t.history[j].frameID, t.history[midx].frameID, 
                                        t.history[end_idx].frameID, t.history_X.shape[0], t.objID]))
            y_new_labels = np.append(y_new_labels, labels[i])
    return np.array(X_feature_vectors), np.array(y_new_labels, dtype=int), np.array(metadata)

def make_feature_vectors_version_seven(trackedObjects: list, labels: np.ndarray, max_stride: int):
    weights = np.array([1,1,100,100,2,2,200,200], dtype=np.float32)
    X_feature_vectors = np.array([])
    y_new_labels = np.array([])
    metadata = []
    for i, t in tqdm.tqdm(enumerate(trackedObjects), desc="Features for classification.", total=len(trackedObjects)):
        stride = max_stride
        if stride > t.history_X.shape[0]:
            continue
        for j in range(0, t.history_X.shape[0]-max_stride, max_stride):
            #midx = j + (3*stride // 4) - 1
            end_idx = j + stride - 1
            feature_vector = np.array([t.history_X[j], t.history_Y[j], t.history_VX_calculated[j], t.history_VY_calculated[j],
                                    t.history_X[end_idx], t.history_Y[end_idx], t.history_VX_calculated[end_idx], t.history_VY_calculated[end_idx]]) * weights
            if X_feature_vectors.shape == (0,):
                X_feature_vectors = np.array(feature_vector).reshape((-1,feature_vector.shape[0]))
            else:
                X_feature_vectors = np.append(X_feature_vectors, np.array([feature_vector]), axis=0)
            metadata.append(np.array([t.history[j].frameID,
                                        t.history[end_idx].frameID, t.history_X.shape[0], t.objID]))
            y_new_labels = np.append(y_new_labels, labels[i])
    return np.array(X_feature_vectors), np.array(y_new_labels, dtype=int), np.array(metadata)

def iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """Generate minibatches for training.

    Args:
        X (np.ndarray): Feature vectors shape(n_samples, n_features) 
        y (np.ndarray): Labels of vectors shape(n_samples,) 
    """
    current_batch_size = batch_size
    X_batch, y_batch = X[:current_batch_size], y[:current_batch_size]
    while X.shape[0] - current_batch_size >= batch_size:
        yield X_batch, y_batch
        X_batch, y_batch = X[:current_batch_size], y[:current_batch_size]
        current_batch_size += batch_size
    else:
        last_batch_size = X.shape[0] % batch_size
        yield X[:last_batch_size], y[:last_batch_size]

def data_preprocessing_for_classifier(path2db: str, min_samples=10, max_eps=0.2, xi=0.1, min_cluster_size=10, n_jobs=18, from_half=False, features_v2=False, features_v2_half=False, features_v3=False):
    """Preprocess database data for classification.
    Load, filter, run clustering on dataset then extract feature vectors from dataset.

    Args:
        path2db (str): Path to database. 
        min_samples (int, optional): Optics Clustering param. Defaults to 10.
        max_eps (float, optional): Optics Clustering param. Defaults to 0.1.
        xi (float, optional): Optics clustering param. Defaults to 0.15.
        min_cluster_size (int, optional): Optics clustering param. Defaults to 10.
        n_jobs (int, optional): Paralell jobs to run. Defaults to 18.

    Returns:
        List[np.ndarray]: X_train, y_train, metadata_train, X_test, y_test, metadata_test, filteredTracks
    """
    from clustering import optics_clustering_on_nx4

    thres = 0.5
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    filteredTracks = filter_trajectories(tracks, threshold=thres)
    filteredTracks = filter_by_class(filteredTracks)
    labels = optics_clustering_on_nx4(filteredTracks, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs, path2db=path2db, threshold=thres)

    if from_half:
        X, y, metadata = make_feature_vectors_version_one_half(filteredTracks, 6, labels)
    elif features_v2:
        X, y, metadata = make_feature_vectors_version_two(filteredTracks, 6, labels)
    elif features_v2_half:
        X, y, metadata = make_feature_vectors_version_two_half(filteredTracks, 6, labels)
    elif features_v3:
        X, y, metadata = make_feature_vectors_version_three(filteredTracks, 6, labels)
    else:
        X, y, metadata = make_feature_vectors_version_one(filteredTracks, 6, labels)

    X = X[y > -1]
    y = y[y > -1]

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    metadata_train = []
    metadata_test = []

    for i in range(len(X)):
        if i%5==0:
            X_test.append(X[i])
            y_test.append(y[i])
            metadata_test.append(metadata[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            metadata_train.append(metadata[i])

    return np.array(X_train), np.array(y_train), np.array(metadata_train), np.array(X_test), np.array(y_test), np.array(metadata_test), filteredTracks

# deprecated
def data_preprocessing_for_calibrated_classifier(path2db: str, min_samples=10, max_eps=0.2, xi=0.1, min_cluster_size=10, n_jobs=18):
    """Preprocess database data for classification.
    Load, filter, run clustering on dataset then extract feature vectors from dataset.

    Args:
        path2db (str): _description_
        min_samples (int, optional): _description_. Defaults to 10.
        max_eps (float, optional): _description_. Defaults to 0.1.
        xi (float, optional): _description_. Defaults to 0.15.
        min_cluster_size (int, optional): _description_. Defaults to 10.
        n_jobs (int, optional): _description_. Defaults to 18.

    Returns:
        List[np.ndarray]: Return X and y train and test dataset 
    """
    from clustering import optics_clustering_on_nx4 
    thres = 0.5
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    filteredTracks = filter_trajectories(tracks, threshold=thres)
    filteredTracks = filter_by_class(filteredTracks)
    labels = optics_clustering_on_nx4(filteredTracks, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, path2db=path2db, threshold=thres, n_jobs=n_jobs, show=True)
    X, y = make_features_for_classification(filteredTracks, 6, labels)
    X = X[y > -1]
    y = y[y > -1]
    X_train = []
    y_train = []
    X_calib = []
    y_calib = []
    X_test = []
    y_test = []
    first_third_limit = int(len(X) * 0.4) 
    second_third_limit = 2*first_third_limit
    for i in range(len(X)):
        if i > second_third_limit-1:
            X_test.append(X[i])
            y_test.append(y[i])
        elif i > first_third_limit-1 and i < second_third_limit-1: 
            X_calib.append(X[i])
            y_calib.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    return np.array(X_train), np.array(y_train), np.array(X_calib), np.array(y_calib), np.array(X_test), np.array(y_test)

def save_model(savedir: str, classifier_type: str, model: OneVSRestClassifierExtended = None):
    """Save model to research_data dir.

    Args:
        path2db (str): Path to database file. 
        classifier_type (str): Classifier name. 
        model (Model): The model itself. 
    """
    if not os.path.isdir(os.path.join(savedir, "models")):
        os.mkdir(os.path.join(savedir, "models"))
    savepath = os.path.join(savedir, "models")
    filename = os.path.join(savepath, f"{classifier_type}.joblib")
    if model is not None:
        joblib.dump(model, filename)
    else:
        print("Error: model is None, model was not saved.")

def load_model(path2model: str) -> OneVSRestClassifierExtended:
    """Load classifier model.

    Args:
        path2model (str): Path to model. 

    Returns:
        BinaryClassifier: Trained binary classifier model. 
    """
    return joblib.load(path2model)

def data_preprocessing_for_classifier_from_joblib_model(model, min_samples=10, max_eps=0.2, xi=0.15, min_cluster_size=10, n_jobs=18, from_half=False, features_v2=False, features_v2_half=False, features_v3=False):
    """Preprocess database data for classification.
    Load, filter, run clustering on dataset then extract feature vectors from dataset.

    Args:
        path2db (str): _description_
        min_samples (int, optional): _description_. Defaults to 10.
        max_eps (float, optional): _description_. Defaults to 0.1.
        xi (float, optional): _description_. Defaults to 0.15.
        min_cluster_size (int, optional): _description_. Defaults to 10.
        n_jobs (int, optional): _description_. Defaults to 18.

    Returns:
        List[np.ndarray]: X_train, y_train, metadata_train, X_test, y_test, metadata_test
    """
    from clustering import optics_on_featureVectors 

    featureVectors = make_4D_feature_vectors(model.tracks)
    labels = optics_on_featureVectors(featureVectors, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, max_eps=max_eps, n_jobs=n_jobs) 

    if from_half:
        X, y, metadata = make_feature_vectors_version_one_half(model.tracks, 6, labels)
    elif features_v2:
        X, y, metadata = make_feature_vectors_version_two(model.tracks, 6, labels)
    elif features_v2_half:
        X, y, metadata = make_feature_vectors_version_two_half(model.tracks, 6, labels)
    elif features_v3:
        X, y, metadata = make_feature_vectors_version_three(model.tracks, 6, labels)
    else:
        X, y, metadata = make_feature_vectors_version_one(model.tracks, 6, labels)

    X = X[y > -1]
    y = y[y > -1]

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    metadata_test = []
    metadata_train = []

    for i in range(len(X)):
        if i%5==0:
            X_test.append(X[i])
            y_test.append(y[i])
            metadata_test.append(metadata[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            metadata_train.append(metadata[i])

    return np.array(X_train), np.array(y_train), np.array(metadata_train), np.array(X_test), np.array(y_test), np.array(metadata_test) 

def preprocess_dataset_for_training(path2dataset: str, min_samples=10, max_eps=0.2, xi=0.15, min_cluster_size=10, n_jobs=18, cluster_features_version: str = "4D", threshold: float = 0.4, classification_features_version: str = "v1", stride: int = 15, level: float = None, n_weights: int = 3, weights_preset: int = 1, p_norm: int = 2):
    from clustering import optics_on_featureVectors 

    tracks = load_dataset(path2dataset)
    tracks = filter_by_class(tracks)
    tracks = filter_trajectories(trackedObjects=tracks, threshold=threshold)

    if cluster_features_version == "4D":
        featureVectors = make_4D_feature_vectors(tracks)
    elif cluster_features_version == "6D":
        featureVectors = make_6D_feature_vectors(tracks)

    labels = optics_on_featureVectors(featureVectors, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, max_eps=max_eps, n_jobs=n_jobs, p=p_norm) 

    if classification_features_version == "v1":
        X, y, metadata = make_feature_vectors_version_one(tracks, 6, labels)
    elif classification_features_version == "v1_half":
        X, y, metadata = make_feature_vectors_version_one_half(tracks, 6, labels)
    elif classification_features_version == "v2":
        X, y, metadata = make_feature_vectors_version_two(tracks, 6, labels)
    elif classification_features_version == "v2_half":
        X, y, metadata = make_feature_vectors_version_two_half(tracks, 6, labels)
    elif classification_features_version == "v3":
        X, y, metadata = make_feature_vectors_version_three(tracks, 6, labels)
    elif classification_features_version == "v3_half":
        X, y, metadata = make_feature_vectors_version_three_half(tracks, 6, labels)
    elif classification_features_version == "v4":
        X, y, metadata = make_feature_vectors_version_four(tracks, stride, labels)
    elif classification_features_version == "v5":
        X, y, metadata = make_feature_vectors_version_five(tracks, labels, stride, n_weights)
    elif classification_features_version == "v6":
        weights_presets = {
            1 : np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0]),
            2 : np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        }
        X, y, metadata = make_feature_vectors_version_five(tracks, labels, stride, weights_presets[weights_preset])
    elif classification_features_version == "v7":
        X, y, metadata = make_feature_vectors_version_seven(tracks, labels, stride)


    X = X[y > -1]
    y = y[y > -1]

    if level is not None:
        X, y = level_features(X, y, level)

    """X_train = []
    y_train = []
    X_test = []
    y_test = []
    metadata_test = []
    metadata_train = []

    for i in range(len(X)):
        if i%5==0:
            X_test.append(X[i])
            y_test.append(y[i])
            metadata_test.append(metadata[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            metadata_train.append(metadata[i])
    """

    return np.array(X), np.array(y), np.array(metadata), tracks, labels

def tracks2joblib(path2db: str, n_jobs=18):
    """Extract tracks from database and save them in a joblib object.

    Args:
        path2db (str): Path to database. 
        n_jobs (int, optional): Paralell jobs to run. Defaults to 18.
    """
    path = Path(path2db)
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs)
    savepath = path.with_suffix(".joblib")
    print('Saving: ', savepath)
    joblib.dump(tracks, savepath)

def trackslabels2joblib(path2tracks: str, output: str, min_samples = 10, max_eps = 0.2, xi = 0.15, min_cluster_size = 10, n_jobs = 18, threshold = 0.5, p = 2, cluster_dimensions: str = "4D"):
    """Save training tracks with class numbers ordered to them.

    Args:
        path2tracks (str): Path to dataset. 
        min_samples (int, optional): Optics clustering parameter. Defaults to 10.
        max_eps (float, optional): Optics clustering parameter. Defaults to 0.2.
        xi (float, optional): Optics clustering parameter. Defaults to 0.15.
        min_cluster_size (int, optional): Optics clustering parameter. Defaults to 10.
        n_jobs (int, optional): Number of processes to run. Defaults to 18.

    Returns:
        _type_: _description_
    """
    from clustering import clustering_on_feature_vectors 
    from sklearn.cluster import OPTICS
    filext = path2tracks.split('/')[-1].split('.')[-1]
    
    if filext == 'db':
        tracks = preprocess_database_data_multiprocessed(path2tracks, n_jobs)
    elif filext == 'joblib':
        tracks = load_dataset(path2tracks)
    else:
        print("Error: Wrong file type.")
        return False
    
    tracks_filtered = filter_trajectories(tracks, threshold=threshold)
    tracks_car_only = filter_by_class(tracks_filtered)

    if cluster_dimensions == "6D":
        cluster_features = make_6D_feature_vectors(tracks_car_only)
    else:
        cluster_features = make_4D_feature_vectors(tracks_car_only)
    _, labels = clustering_on_feature_vectors(cluster_features, OPTICS,
                                            n_jobs=n_jobs,
                                            p=p,
                                            min_samples=min_samples, 
                                            max_eps=max_eps, 
                                            xi=xi, 
                                            min_cluster_size=min_cluster_size)
    
    # order labels to tracks, store it in a list[dictionary] format
    tracks_classes = []
    for i, t in enumerate(tracks_car_only):
        tracks_classes.append({
            "track": t,
            "class": labels[i] 
        })

    #filename = path2tracks.split('/')[-1].split('.')[0] + '_clustered.joblib' 
    #avepath = os.path.join("research_data", path2tracks.split('/')[-1].split('.')[0], filename) 

    print("Saving: ", output)

    return joblib.dump(tracks_classes, output, compress="lz4")

def load_dataset(path2dataset: str | Path | list[str]):
    """Load dataset from either a joblib file or a database file.
    If dataset path is a directory load all joblib files from the directory.
    dict['track': TrackedObject, 'class': label].

    Args:
        path2dataset (str): Path to file containing dataset. 

    Returns:
        list[TrackedObject]: list of TrackedObject objects. 
    """
    if type(path2dataset) == list:
        datasets = []
        for p in path2dataset:
            datasets.append(load_dataset(p))
        return mergeDatasets(datasets) 
    datasetPath = Path(path2dataset)
    ext = datasetPath.suffix
    if ext == ".joblib":
        dataset = joblib.load(path2dataset)
        if type(dataset[0]) == dict:
            ret_dataset = [d['track'] for d in dataset] 
            return ret_dataset
        else:
            return np.array(dataset)
    elif ext == ".db":
        return np.array(preprocess_database_data_multiprocessed(path2dataset, n_jobs=None))
    elif Path.is_dir(datasetPath):
        return mergeDatasets(loadDatasetsFromDirectory(datasetPath))
    elif type(path2dataset) == list:
        tmp_dataset = []
        for d in path2dataset:
            tmp_dataset.append(load_dataset(d))
        tmp_dataset = np.array(tmp_dataset)
        return mergeDatasets(tmp_dataset)
    raise Exception("Wrong file type.")

def mergeDatasets(datasets: np.ndarray):
    """Merge datasets into one.

    Args:
        datasets (ndarray): List of datasets to merge. 
            shape(n, m) where n is the number of datasets 
            and m is the number of tracks in the dataset.
    
    Returns:
        ndarray: Merged dataset.
    """
    merged = np.array([])
    for d in datasets:
        merged = np.append(merged, d)
    return merged

def strfy_dict_params(params: dict):
    """Stringify params stored in dictionaries.

    Args:
        params (dict): Dict storing the params. 

    Returns:
        str: Stringified params returned in the format "_param1_value1_param2_value2". 
    """
    ret_str = ""
    if len(params) == 0:
        return ret_str
    for p in params:
        ret_str += str("_"+p+"_"+str(params[p]))
    return ret_str
    
def downscale_TrackedObjects(trackedObjects: list, img: np.ndarray):
    """Normalize the values of the detections with the given np.ndarray image.

    Args:
        trackedObjects (list[TrackedObject]): list of tracked objects 
        img (np.ndarray): image to downscale from 
    """
    ret_trackedObjects = []
    aspect_ratio = img.shape[1] / img.shape[0]
    for o in tqdm.tqdm(trackedObjects, desc="Downscale"):
        t = deepcopy(o)
        t.history_X = t.history_X / img.shape[1] * aspect_ratio 
        t.history_Y = t.history_Y / img.shape[0]
        t.history_VX_calculated = t.history_VX_calculated / img.shape[1] * aspect_ratio 
        t.history_VY_calculated = t.history_VY_calculated / img.shape[0]
        t.history_AX_calculated = t.history_AX_calculated / img.shape[1] * aspect_ratio 
        t.history_AY_calculated = t.history_AY_calculated / img.shape[0]
        for d in t.history:
            d.X = d.X / img.shape[1] * aspect_ratio
            d.Y = d.Y / img.shape[0]
            d.VX = d.VX / img.shape[1] * aspect_ratio
            d.VY = d.VY / img.shape[0]
            d.AX = d.AX / img.shape[1] * aspect_ratio
            d.AY = d.AY / img.shape[0]
            d.Width = d.Width / img.shape[1] * aspect_ratio
            d.Height = d.Height / img.shape[0]
        ret_trackedObjects.append(t)
    return ret_trackedObjects

def diff(x_1: float, x_2: float, dt: float) -> float:
    """Differentiate with function x_(i+1) - x_i / dt

    Args:
        x_1 (float): x_i 
        x_2 (float): x_(i+1) 
        dt (float): dt 

    Returns:
        float: dx
    """
    if dt == 0:
        return 0
    return (x_2-x_1) / dt 

def dt(t1: float, t2: float) -> float:
    """Calculate dt

    Args:
        t1 (float): t_i 
        t2 (float): t_(i+1) 

    Returns:
        float: dt 
    """
    return t2-t1

def diffmap(a: np.array, t: np.array, k: int):
    """Differentiate an array `a` with time vector `t`, and `k` i+k in the function x_(i+k) - x_i / t_(i+k) - t_i

    Args:
        a (np.array): array of values to differentiate 
        t (np.array): times to differentiate with 
        k (int): stepsize 

    Returns:
        np.array, np.array: Return dX and t timestamps of dX with the logic dx_i, t_i+k 
    """
    X = np.array([])
    T = np.array([])
    if a.shape[0] < k:
        for i in range(a.shape[0]-1):
            T = np.append(T, [t[i]])
            X = np.append(X, [0])
    else:
        for i in range(0, k-1):
            T = np.append(T, [t[i]])
            X = np.append(X, [0])
        for i in range(k, a.shape[0]):
            dt_ = dt(t[i], t[i-k])
            T = np.append(T, t[i])
            X = np.append(X, diff(a[i], a[i-k], dt_))
    return X, T 

def trackedObjects_old_to_new(trackedObjects: list, k_velocity: int = 10, k_accel: int = 2):
    """Depracated function. Archived.

    Args:
        trackedObjects (list[TrackedObject]): tracked objects. 

    Returns:
        list: list of converted tracked objects 
    """
    new_trackedObjects = []
    for t in tqdm.tqdm(trackedObjects, desc="TrackedObjects converted to new class structure."):
        tmp_obj = dataManagementClasses.TrackedObject(t.objID, t.history[0])
        tmp_obj.history = t.history
        tmp_obj.history_X = np.array([d.X for d in t.history])
        tmp_obj.history_Y = np.array([d.Y for d in t.history])
        tmp_obj.X = t.X
        tmp_obj.Y = t.Y
        tmp_obj.VX = t.VX
        tmp_obj.VY = t.VY
        tmp_obj.AX = t.AX
        tmp_obj.AY = t.AY
        tmp_obj.futureX = t.futureX
        tmp_obj.futureY = t.futureY
        tmp_obj.isMoving = t.isMoving
        tmp_obj.label = t.label
        T = np.array([d.frameID for d in tmp_obj.history])
        tmp_obj.history_VX_calculated, tmp_obj.history_VT= diffmap(tmp_obj.history_X, T, k_velocity)
        tmp_obj.history_VY_calculated, _ = diffmap(tmp_obj.history_Y, T, k_velocity)
        tmp_obj.history_AX_calculated, _ = diffmap(tmp_obj.history_VX_calculated, tmp_obj.history_VT, k_accel)
        tmp_obj.history_AY_calculated, _ = diffmap(tmp_obj.history_VY_calculated, tmp_obj.history_VT, k_accel)
        tmp_obj.history_VX_calculated = np.insert(tmp_obj.history_VX_calculated, 0, [0])
        tmp_obj.history_VY_calculated = np.insert(tmp_obj.history_VY_calculated, 0, [0])
        tmp_obj.history_AX_calculated = np.insert(tmp_obj.history_AX_calculated, 0, [0,0])
        tmp_obj.history_AY_calculated = np.insert(tmp_obj.history_AY_calculated, 0, [0,0])
        new_trackedObjects.append(tmp_obj)
    return new_trackedObjects

def level_features(X: np.ndarray, y: np.ndarray, ratio_to_min: float = 2.0):
    """Level out the nuber of features.

    Args:
        X (np.ndarray): features of shape(n_samples, n_features) 
        y (np.ndarray): labels of shape(n_samples,) 

    Raises:
        ValueError: If the length of axis 0 of both X and y are not equal raise ValueError. 

    Returns:
        np.ndarray, np.ndarray: Numpy array of leveled out X and y.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples and number of labels should be equal.")
    labels = list(set(y))
    label_counts = np.zeros(shape=(len(labels)), dtype=int) # init counter vector
    y = y.astype(int)
    for y_ in y:
        label_counts[y_] += 1
    # find min count
    #min_sample_count = np.min(label_counts) 
    min_sample_label = np.argmin(label_counts)
    # init X and y vectors that will be filled and returned
    X_leveled = np.array([], dtype=float) 
    y_leveled = np.array([], dtype=int)
    print(labels)
    print(label_counts)
    new_label_counts = np.array([])
    for l in tqdm.tqdm(labels):
        i = 0
        j = 0
        if l != min_sample_label:
            sample_limit = int(ratio_to_min * label_counts[min_sample_label])
        else:
            sample_limit = label_counts[min_sample_label]
        new_label_counts = np.append(new_label_counts, [sample_limit])
        while i < sample_limit and i < label_counts[int(l)]:
            if y[j] == l:
                if X_leveled.shape == (0,):
                    X_leveled = np.array([X[j]], dtype=float) 
                else:
                    X_leveled = np.append(X_leveled, [X[j]], axis=0)
                y_leveled = np.append(y_leveled, [y[j]])
                i+=1
            j+=1
    print(labels)
    print(new_label_counts)
    return X_leveled, y_leveled

def upscale_cluster_centers(centroids, framewidth: int, frameheight: int):
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
        retarr[i, 0] = (centroids[i, 0] * framewidth) / ratio
        retarr[i, 1] = centroids[i, 1] * frameheight
    return retarr

def calc_cluster_centers(tracks, labels, exit = True):
    """Calculate center of mass for every class's exit points.
    These will be the exit points of the clusters.

    Args:
        tracks (List[TrackedObject]): List of tracked objects 
        labels (_type_): Labels of tracked objects 

    Returns:
        cluster_centers: The center of mass for every class's exits points
    """
    classes = set(labels)
    cluster_centers = np.zeros(shape=(len(classes),2))
    for i, c in enumerate(classes):
        tracks_xy = []
        for j, l in enumerate(labels):
            if l==c:
                if exit:
                    tracks_xy.append([tracks[j].history[-1].X, tracks[j].history[-1].Y])
                else:
                    tracks_xy.append([tracks[j].history[0].X, tracks[j].history[0].Y])
        tracks_xy = np.array(tracks_xy)
        cluster_centers[i,0] = np.average(tracks_xy[:,0])
        cluster_centers[i,1] = np.average(tracks_xy[:,1])
    return cluster_centers

def loadDatasetsFromDirectory(path):
    """Load all datasets from a directory.

    Args:
        path (str | Path): Directory path. 

    Returns:
        ndarray: Numpy array of all datasets. 
    """
    dirPath = Path(path)
    if not dirPath.is_dir():
        return False
    dataset = np.array([], dtype=dataManagementClasses.TrackedObject)
    for p in dirPath.glob("*.joblib"):
        tmpDataset = load_dataset(p)
        dataset = np.append(dataset, tmpDataset, axis=0)
        print(len(tmpDataset))
    return dataset

def load_dataset_with_labels(path):
    dataset = load_dataset(path)
    dataset_labeled = (dataset, str(path))
    return dataset_labeled

def loadDatasetMultiprocessedCallback(result):
    print(len(result[0]), result[1])

def loadDatasetMultiprocessed(path, n_jobs=-1):
    from multiprocessing import Pool
    dirPath = Path(path)
    if not dirPath.is_dir():
        return False
    datasetPaths = [p for p in dirPath.glob("*.joblib")]
    dataset = []
    with Pool(processes=n_jobs) as pool:
        for i, p in enumerate(datasetPaths):
            tmpDatasetLabeled = pool.apply_async(load_dataset_with_labels, (p,), callback=loadDatasetMultiprocessedCallback)
            dataset.append(tmpDatasetLabeled.get())
    return np.array(dataset)

def plot_misclassified(misclassifiedTracks: List[dataManagementClasses.TrackedObject], output: str = None):
    """Plot misclassified trajectories. If output is given, save plot to output/plots/misclassified.png.

    Args:
        misclassifiedTracks (List[dataManagementClasses.TrackedObject]): List of TrackedObjects, which was misclassified by an estimator.
        output (str, optional): Output directory path. Defaults to None.
    """
    X_enter = [t.history_X[0] for t in misclassifiedTracks]
    Y_enter = [t.history_Y[0] for t in misclassifiedTracks]
    X_exit = [t.history_X[-1] for t in misclassifiedTracks]
    Y_exit = [t.history_Y[-1] for t in misclassifiedTracks]
    X_traj = np.ravel([t.history_X[1:-1] for t in misclassifiedTracks])
    Y_traj = np.ravel([t.history_Y[1:-1] for t in misclassifiedTracks])
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(X_enter, Y_enter, s=10, c='g')
    ax.scatter(X_exit, Y_exit, s=10, c='r')
    ax.scatter(X_traj, Y_traj, s=5, c='b')
    fig.show()
    if output is not None:
        _output = Path(output) / "plots"
        _output.mkdir(exist_ok=True)
        fig.savefig(fname=(_output / "misclassified.png"))

def plot_misclassified_feature_vectors(misclassifiedFV: np.ndarray, output: str = None, background: str = None, classifier: str = "SVM"):
    """Plot misclassified trajectories. If output is given, save plot to output/plots/misclassified.png.

    Args:
        misclassifiedTracks (List[dataManagementClasses.TrackedObject]): List of TrackedObjects, which was misclassified by an estimator.
        output (str, optional): Output directory path. Defaults to None.
        background (str, optional): Background image of plot for better visualization.
    """
    X_mask = [False, False, False, False, False, False, True, False, False, False]
    Y_mask = [False, False, False, False, False, False, False, True, False, False]
    X = np.ravel([f[X_mask] for f in misclassifiedFV])
    Y = np.ravel([f[Y_mask] for f in misclassifiedFV])
    fig, ax = plt.subplots(figsize=(7,7))
    if background is not None:
        print(background)
        I = plt.imread(fname=background)
        # I = plt.imread("/media/pecneb/4d646cbd-cce0-42c4-bdf5-b43cc196e4a1/gitclones/computer_vision_research/research_data/Bellevue_150th_Newport_24h_v2/Preprocessed/Bellevue_150th_Newport.JPG", format="jpg")
        ax.imshow(I, alpha=0.4, extent=[0, 1280, 0, 720])
    ax.scatter((X * I.shape[1]) / (I.shape[1] / I.shape[0]), (1-Y) * I.shape[0], s=0.05, c='r')
    ax.set_xlim(left=0, right=1280)
    ax.set_ylim(bottom=0, top=720)
    ax.grid(visible=True)
    ax.set_title(label=f"{classifier} misclassifications")
    fig.show()
    if output is not None:
        _output = Path(output) / "plots"
        _output.mkdir(exist_ok=True)
        fig.savefig(fname=(_output / f"{classifier}_misclassified.png"))

def save_trajectories(trajectories: List[dataManagementClasses.TrackedObject] or np.ndarray, output: str or Path, classifier: str = "SVM") -> List[str]:
    _output = Path(output)
    _outdir = _output / "misclassified_trajectories"
    _outdir.mkdir(exist_ok=True)
    _filename = _outdir / f"{classifier}_miclassified_trajectories.joblib"
    return joblib.dump(trajectories, filename=_filename)