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
import matplotlib.pyplot as plt
from dataManagementClasses import Detection, TrackedObject
import numpy as np
import time
import databaseLoader
import tqdm
import os
import joblib 

def savePlot(fig: plt.Figure, name: str):
    fig.savefig(name, dpi=150)

def detectionFactory(objID: int, frameNum: int, label: str, confidence: float, x: float, y: float, width: float, height: float, vx: float, vy: float, ax:float, ay: float) -> Detection:
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
        ax (float): Accelaration on the X axis. 
        ay (float): Accelaration on the Y axis. 

    Returns:
        Detection: The Detection object, which is to be returned. 
    """
    retDet = Detection(label, confidence, x,y,width,height,frameNum)
    retDet.objID = objID
    retDet.VX = vx
    retDet.VY = vy
    retDet.AX = ax
    retDet.AY = ay
    return retDet

def trackedObjectFactory(detections: list) -> TrackedObject:
    """Create trackedObject object from list of detections

    Args:
        detections (list): list of detection 

    Returns:
        TrackedObject:  trackedObject
    """
    tmpObj = TrackedObject(detections[0].objID, detections[0], len(detections))
    for det in detections:
        tmpObj.history.append(det)
        tmpObj.X = det.X
        tmpObj.Y = det.Y
        tmpObj.VX = det.VX
        tmpObj.VY = det.VY
        tmpObj.AX = det.AX
        tmpObj.AY = det.AY
    return tmpObj

def cvCoord2npCoord(Y: np.ndarray) -> np.ndarray:
    """Convert OpenCV Y axis coordinates to numpy coordinates.

    Args:
        Y (np.ndarray): Y axis coordinate vector

    Returns:
        np.ndarray: Y axis coordinate vector
    """
    return 1 - Y

def detectionParser(rawDetectionData) -> list:
    """Parse raw detection data loaded from database.
    Returns a list of detections. 

    Args:
        rawDetectionData (list): list of raw detection entries

    Returns:
        detections: list of parsed detections 
    """
    detections = []
    for entry in rawDetectionData:
        detections.append(detectionFactory(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7], entry[8], entry[9], entry[10], entry[11]))
    return detections 

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
        return trackedObjectFactory(detectionParser(rawDets))
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
    

def filter_out_false_positive_detections(trackedObjects: list, threshold: float):
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

def filter_out_edge_detections(trackedObjects: list, threshold: float):
    """Filter out objects, that enter and exit detections coordinates are in the threshold value.

    Args:
        trackedObjects (list): list of object trackings 
        threshold (float): objects only under this value will be returned 

    Returns:
        list: filtered list of tracks 
    """
    max_y = 0 
    min_y = 9999 
    max_x = 0
    min_x = 9999
    for obj in tqdm.tqdm(trackedObjects, desc="Looking for min max values."):
        local_min_x = np.min([det.X for det in obj.history])
        local_max_x = np.max([det.X for det in obj.history])
        local_min_y = np.min([det.Y for det in obj.history])
        local_max_y = np.max([det.Y for det in obj.history])
        if max_x < local_max_x:
            max_x = local_max_x
        if min_x > local_min_x:
            min_x = local_min_x
        if max_y < local_max_y:
            max_y = local_max_y
        if min_y > local_min_y:
            min_y = local_min_y 
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
    return filter_out_false_positive_detections(filteredTracks, 0.4)

def filter_tracks(trackedObjects: list, label="car"):
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

def makeFeatureVectorsNx4(trackedObjects: list) -> np.ndarray:
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

def make_features_for_classification_velocity_time(trackedObjects: list, k: int, labels: np.ndarray):
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
    for j in range(len(trackedObjects)):
        step = len(trackedObjects[j].history)//k
        if step > 0:
            midstep = step//2
            for i in range(0, len(trackedObjects[j].history)-step, step):
                featureVectors.append(np.array([trackedObjects[j].history[i].X, trackedObjects[j].history[i].Y, 
                                            trackedObjects[j].history[i].VX, trackedObjects[j].history[i].VY,
                                            trackedObjects[j].history[i+midstep].X, trackedObjects[j].history[i+midstep].Y,
                                            trackedObjects[j].history[i+step].X, trackedObjects[j].history[i+step].Y,
                                            trackedObjects[j].history[i+step].VX, trackedObjects[j].history[i+step].VY]))
                newLabels.append(labels[j])
                track_history_metadata.append([trackedObjects[j].history[i].frameID, trackedObjects[j].history[i+midstep].frameID, 
                trackedObjects[j].history[i+step].frameID, len(trackedObjects[j].history), trackedObjects[j].objID])
    return np.array(featureVectors), np.array(newLabels), np.array(track_history_metadata)

def make_features_for_classification_velocity_time_second_half(trackedObjects: list, k: int, labels: np.ndarray):
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
    for j in range(len(trackedObjects)):
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
    for i, track in enumerate(trackedObjects):
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
    for i, track in enumerate(trackedObjects):
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
    this last detection reaches the end of the history. Next to the coordinates, also the velocity of the object is being included in the feature vector.

    Args:
        trackedObjects (list): Tracked objects. 
        labels (np.ndarray): Labels of the tracks, which belongs to a given cluster, given by the clustering algo. 

    Returns:
        tuple of numpy arrays: The newly created feature vectors, the labels created for each feature vector, and the metadata that contains the information of time frames, and to which object does the feature belongs to. 
    """
    from visualizer import aoiextraction
    X_featurevectors = []
    y_newLabels = []
    cluster_centroids = aoiextraction(trackedObjects, labels)
    featurevector_metadata = [] # [start_time, mid_time, end_time, history_length, trackID]
    for i in range(len(trackedObjects)):
        step = (len(trackedObjects[i].history))//k
        if step >= 2:
            for j in range(step, len(trackedObjects[i].history), step):
                midx = j//2
                """ X_featurevectors.append(np.array([trackedObjects[i].history[0].X, trackedObjects[i].history[0].Y, 
                                                #trackedObjects[i].history[0].VX, trackedObjects[i].history[0].VY, 
                                                trackedObjects[i].history[midx].X, trackedObjects[i].history[midx].Y, 
                                                trackedObjects[i].history[j].X, trackedObjects[i].history[j].Y, 
                                                #trackedObjects[i].history[j].VX, trackedObjects[i].history[j].VY
                                                cluster_centroids[labels[i]][0] - trackedObjects[i].history[0].X, 
                                                cluster_centroids[labels[i]][1] - trackedObjects[i].history[0].Y, 
                                                cluster_centroids[labels[i]][0] - trackedObjects[i].history[midx].X, 
                                                cluster_centroids[labels[i]][1] - trackedObjects[i].history[midx].Y,
                                                cluster_centroids[labels[i]][0] - trackedObjects[i].history[j].X, 
                                                cluster_centroids[labels[i]][1] - trackedObjects[i].history[j].Y,])) """
                fv = np.array([
                            trackedObjects[i].history[0].X, trackedObjects[i].history[0].Y, 
                            #trackedObjects[i].history[0].VX, trackedObjects[i].history[0].VY, 
                            trackedObjects[i].history[midx].X, trackedObjects[i].history[midx].Y, 
                            trackedObjects[i].history[j].X, trackedObjects[i].history[j].Y, 
                            #trackedObjects[i].history[j].VX, trackedObjects[i].history[j].VY
                            ])
                """for c in cluster_centroids:
                    fv = np.append(fv, [
                                        cluster_centroids[c][0] - trackedObjects[i].history[0].X, 
                                        cluster_centroids[c][1] - trackedObjects[i].history[0].Y, 
                                        cluster_centroids[c][0] - trackedObjects[i].history[midx].X, 
                                        cluster_centroids[c][1] - trackedObjects[i].history[midx].Y,
                                        cluster_centroids[c][0] - trackedObjects[i].history[j].X, 
                                        cluster_centroids[c][1] - trackedObjects[i].history[j].Y,
                                        ]
                                    )"""
                X_featurevectors.append(fv)
                y_newLabels.append(labels[i])
                featurevector_metadata.append(np.array([trackedObjects[i].history[0].frameID, trackedObjects[i].history[midx].frameID, 
                                            trackedObjects[i].history[j].frameID, len(trackedObjects[i].history), trackedObjects[i].objID]))
    return np.array(X_featurevectors), np.array(y_newLabels), np.array(featurevector_metadata)

def make_feature_vectors_version_three_half(trackedObjects: list, k: int, labels: np.ndarray):
    """Make feature vectors from track histories, such as starting from the first detection incrementing the vectors length by a given factor, building multiple vectors from one history.
    A vector is made up from the absolute first detection of the history, a relative middle detection, and a last detecion, that's index is incremented, for the next feature vector until 
    this last detection reaches the end of the history. Next to the coordinates, also the velocity of the object is being included in the feature vector.

    Args:
        trackedObjects (list): Tracked objects. 
        labels (np.ndarray): Labels of the tracks, which belongs to a given cluster, given by the clustering algo. 

    Returns:
        tuple of numpy arrays: The newly created feature vectors, the labels created for each feature vector, and the metadata that contains the information of time frames, and to which object does the feature belongs to. 
    """
    from visualizer import aoiextraction
    X_featurevectors = []
    y_newLabels = []
    cluster_centroids = aoiextraction(trackedObjects, labels)
    featurevector_metadata = [] # [start_time, mid_time, end_time, history_length, trackID]
    for i in range(len(trackedObjects)):
        step = (len(trackedObjects[i].history))//k
        if step >= 2:
            for j in range((len(trackedObjects[i].history)//2)+step, len(trackedObjects[i].history), step):
                midx = j//2
                X_featurevectors.append(np.array([trackedObjects[i].history[0].X, trackedObjects[i].history[0].Y, 
                                                #trackedObjects[i].history[0].VX, trackedObjects[i].history[0].VY, 
                                                trackedObjects[i].history[midx].X, trackedObjects[i].history[midx].Y, 
                                                trackedObjects[i].history[j].X, trackedObjects[i].history[j].Y, 
                                                #trackedObjects[i].history[j].VX, trackedObjects[i].history[j].VY
                                                #cluster_centroids[labels[i]][0] - trackedObjects[i].history[0].X, 
                                                #cluster_centroids[labels[i]][1] - trackedObjects[i].history[0].Y, 
                                                #cluster_centroids[labels[i]][0] - trackedObjects[i].history[midx].X, 
                                                #cluster_centroids[labels[i]][1] - trackedObjects[i].history[midx].Y,
                                                #cluster_centroids[labels[i]][0] - trackedObjects[i].history[j].X, 
                                                #cluster_centroids[labels[i]][1] - trackedObjects[i].history[j].Y,
                                                ]))
                y_newLabels.append(labels[i])
                featurevector_metadata.append(np.array([trackedObjects[i].history[0].frameID, trackedObjects[i].history[midx].frameID, 
                                            trackedObjects[i].history[j].frameID, len(trackedObjects[i].history), trackedObjects[i].objID]))
    return np.array(X_featurevectors), np.array(y_newLabels), np.array(featurevector_metadata)

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
    filteredTracks = filter_out_edge_detections(tracks, threshold=thres)
    filteredTracks = filter_tracks(filteredTracks)
    labels = optics_clustering_on_nx4(filteredTracks, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs, path2db=path2db, threshold=thres)

    if from_half:
        X, y, metadata = make_features_for_classification_velocity_time_second_half(filteredTracks, 6, labels)
    elif features_v2:
        X, y, metadata = make_feature_vectors_version_two(filteredTracks, 6, labels)
    elif features_v2_half:
        X, y, metadata = make_feature_vectors_version_two_half(filteredTracks, 6, labels)
    elif features_v3:
        X, y, metadata = make_feature_vectors_version_three(filteredTracks, 6, labels)
    else:
        X, y, metadata = make_features_for_classification_velocity_time(filteredTracks, 6, labels)

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
    filteredTracks = filter_out_edge_detections(tracks, threshold=thres)
    filteredTracks = filter_tracks(filteredTracks)
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

from classifier import OneVSRestClassifierExtended 
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

    featureVectors = makeFeatureVectorsNx4(model.tracks)
    labels = optics_on_featureVectors(featureVectors, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, max_eps=max_eps, n_jobs=n_jobs) 

    if from_half:
        X, y, metadata = make_features_for_classification_velocity_time_second_half(model.tracks, 6, labels)
    elif features_v2:
        X, y, metadata = make_feature_vectors_version_two(model.tracks, 6, labels)
    elif features_v2_half:
        X, y, metadata = make_feature_vectors_version_two_half(model.tracks, 6, labels)
    elif features_v3:
        X, y, metadata = make_feature_vectors_version_three(model.tracks, 6, labels)
    else:
        X, y, metadata = make_features_for_classification_velocity_time(model.tracks, 6, labels)

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

def preprocess_dataset_for_training(path2dataset: str, min_samples=10, max_eps=0.2, xi=0.15, min_cluster_size=10, n_jobs=18, from_half=False, features_v2=False, features_v2_half=False, features_v3=False, features_v3_half=False):
    from clustering import optics_on_featureVectors 
    from visualizer import aoiextraction

    tracks = load_dataset(path2dataset)

    featureVectors = makeFeatureVectorsNx4(tracks)
    labels = optics_on_featureVectors(featureVectors, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, max_eps=max_eps, n_jobs=n_jobs) 

    if from_half:
        X, y, metadata = make_features_for_classification_velocity_time_second_half(tracks, 6, labels)
    elif features_v2:
        X, y, metadata = make_feature_vectors_version_two(tracks, 6, labels)
    elif features_v2_half:
        X, y, metadata = make_feature_vectors_version_two_half(tracks, 6, labels)
    elif features_v3:
        X, y, metadata = make_feature_vectors_version_three(tracks, 6, labels)
    elif features_v3_half:
        X, y, metadata = make_feature_vectors_version_three_half(tracks, 6, labels)
    else:
        X, y, metadata = make_features_for_classification_velocity_time(tracks, 6, labels)

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

def tracks2joblib(path2db: str, n_jobs=18):
    """Extract tracks from database and save them in a joblib object.

    Args:
        path2db (str): Path to database. 
        n_jobs (int, optional): Paralell jobs to run. Defaults to 18.
    """
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs)
    filename =  path2db.split('/')[-1].split('.')[0] + '.joblib'
    savepath = os.path.join('research_data', path2db.split('/')[-1].split('.')[0])
    print('Saving: ', os.path.join(savepath, filename))
    joblib.dump(tracks, os.path.join(savepath, filename))

def load_joblib_tracks(path2tracks: str):
    """Load tracks from joblib file.

    Args:
        path2tracks (str): Path to joblib file. 

    Returns:
        list[TrackedObjects]: Loaded list of tracked objects. 
    """
    if path2tracks.split('.')[-1] != "joblib":
        print("Error: Not joblib file.")
        exit(1)
    return joblib.load(path2tracks)

def trackslabels2joblib(path2tracks: str, min_samples = 10, max_eps = 0.2, xi = 0.15, min_cluster_size = 10, n_jobs = 18):
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
    from clustering import optics_on_featureVectors 
    filext = path2tracks.split('/')[-1].split('.')[-1]
    
    if filext == 'db':
        tracks = preprocess_database_data_multiprocessed(path2tracks, n_jobs)
    elif filext == 'joblib':
        tracks = load_joblib_tracks(path2tracks)
    else:
        print("Error: Wrong file type.")
        return False
    
    tracks_filtered = filter_out_edge_detections(tracks, 0.5)
    tracks_car_only = filter_tracks(tracks_filtered)

    cluster_features = makeFeatureVectorsNx4(tracks_car_only)
    labels = optics_on_featureVectors(cluster_features, min_samples=min_samples, 
                                    max_eps=max_eps, xi=xi, 
                                    min_cluster_size=min_cluster_size, n_jobs=n_jobs)
    
    # order labels to tracks, store it in a list[dictionary] format
    tracks_classes = []
    for i, t in enumerate(tracks_car_only):
        tracks_classes.append({
            "track": t,
            "class": labels[i] 
        })

    filename = path2tracks.split('/')[-1].split('.')[0] + '_filtered.joblib' 
    savepath = os.path.join("research_data", path2tracks.split('/')[-1].split('.')[0], filename) 

    print("Saving: ", savepath)

    return joblib.dump(tracks_classes, savepath)

def random_split_tracks(dataset: list, train_percentage: float, seed: int):
    """Shuffle track dataset, then split it into a train and test dataset.

    Args:
        dataset (list): Tracked object list. 
        train_percentage (float): What percentage of the dataset should be train dataset. Value between 0.0 - 1.0 
        seed (int): A seed to be able to repeat the shuffle algorithm. 

    Returns:
        tuple(list, list): train, test datasets 
    """
    from sklearn.utils import shuffle

    # calculate train and test dataset size based on the given percentage
    train_size = int(len(dataset) * train_percentage) 
    test_size = len(dataset) - train_size


    train = shuffle(dataset, random_state=seed, n_samples=train_size) 
    test = []

    # fill test dataset with the rest of the tracks
    i = 0
    while(len(test) != test_size or i > len(dataset)):
        if dataset[i] not in train and dataset[i] not in test:
            test.append(dataset[i])
        i += 1

    return train, test

def load_dataset(path2dataset: str):
    """This function loads the track data from sqlite db or joblib file.
    The extract_tracks_from_db.py script can create two type of joblib
    files. One with all the tracks unfiltered, and one with filtered tracks
    that only contain tracks used in clustering an classifier training.
    The filtered dataset stores the tracks with their corresponding labels
    that are assigned at clustering in dictionaries in the following format
    dict['track': TrackedObject, 'class': label].

    Args:
        path2dataset (str): Path to file containing dataset. 

    Returns:
        list[TrackedObject]: list of TrackedObject objects. 
    """
    ext = path2dataset.split('.')[-1] # extension of dataset file
    if ext == "joblib":
        dataset = load_joblib_tracks(path2dataset)
        if len(dataset[0]) == 1:
            return dataset
        elif len(dataset[0]) == 2:
            ret_dataset = [d['track'] for d in dataset] 
            return ret_dataset
    elif ext == "db":
        return preprocess_database_data_multiprocessed(path2dataset, n_jobs=None) # None for n_jobs to utilize all cpu threads
    print("Error: Bad file type.")
    return False 

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
    