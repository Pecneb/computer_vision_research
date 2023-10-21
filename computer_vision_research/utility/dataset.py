import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import List

import joblib
import numpy as np
import tqdm
from .dataManagementClasses import (Detection, TrackedObject, detectionFactory,
                                   trackedObjectFactory)

from . import databaseLoader


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

def load_dataset(path2dataset: str or Path or List[str]):
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
            dataset = ret_dataset
        for d in dataset:
            d._dataset = path2dataset
        return np.array(dataset)
    elif ext == ".db":
        return np.array(preprocess_database_data_multiprocessed(path2dataset, n_jobs=None))
    elif Path.is_dir(datasetPath):
        return mergeDatasets(loadDatasetsFromDirectory(datasetPath))
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
    dataset = np.array([], dtype=TrackedObject)
    for p in dirPath.glob("*.joblib"):
        tmpDataset = load_dataset(p)
        dataset = np.append(dataset, tmpDataset, axis=0)
        # print(len(tmpDataset))
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

def save_trajectories(trajectories: List[TrackedObject] or np.ndarray, output: str or Path, classifier: str = "SVM", name: str = "trajectories") -> List[str]:
    _filename = Path(output) / f"{classifier}_{name}.joblib"
    return joblib.dump(trajectories, filename=_filename)

