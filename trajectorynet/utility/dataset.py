import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import tqdm

from . import databaseLoader


memory = joblib.Memory(location="cache", verbose=0)


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
        t.history_VX_calculated = t.history_VX_calculated / \
            img.shape[1] * aspect_ratio
        t.history_VY_calculated = t.history_VY_calculated / img.shape[0]
        t.history_AX_calculated = t.history_AX_calculated / \
            img.shape[1] * aspect_ratio
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


def loadDatasetsFromDirectory(path: Union[str, Path]) -> Union[np.ndarray, bool]:
    """Load all datasets from a directory.

    Parameters
    ----------
    path : Union[str, Path]
        Path to directory containing datasets.

    Returns
    -------
    Union[np.ndarray, bool]
        Numpy array containing all datasets, or False if path is not a directory.
    """
    dirPath = Path(path)
    if not dirPath.is_dir():
        return False
    dataset = np.array([], dtype=object)
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
            tmpDatasetLabeled = pool.apply_async(
                load_dataset_with_labels, (p,), callback=loadDatasetMultiprocessedCallback)
            dataset.append(tmpDatasetLabeled.get())
    return np.array(dataset)


def save_trajectories(trajectories: Union[List, np.ndarray], output: Union[str, Path], classifier: str = "SVM", name: str = "trajectories") -> List[str]:
    """Save trajectories to a file.

    Parameters
    ----------
    trajectories : Union[List, np.ndarray]
        Trajectories to save.
    output : Union[str, Path]
        Output directory path.
    classifier : str, optional
        Name of classifier, by default "SVM"
    name : str, optional
        Additional name to identify file, by default "trajectories"

    Returns
    -------
    List[str]
        List of saved file paths.

    """
    _filename = Path(output) / f"{classifier}_{name}.joblib"
    return joblib.dump(trajectories, filename=_filename)


def load_dataset(path2dataset: Union[str, List[str], Path]) -> np.ndarray:
    """Load a dataset from a file or a directory.

    Parameters
    ----------
    path2dataset : Union[str, List[str], Path]


    Returns
    -------
    np.ndarray
        Numpy array containing the dataset. 

    Raises
    ------
    IOError
        Wrong file type.

    """
    if type(path2dataset) == list:
        datasets = []
        for p in path2dataset:
            datasets.append(load_dataset(p))
        return mergeDatasets(datasets)
    datasetPath = Path(path2dataset)
    ext = datasetPath.suffix
    if ext == ".joblib":
        try:
            dataset = joblib.load(path2dataset)
        except:
            return np.array([])
        if type(dataset[0]) == dict:
            ret_dataset = [d['track'] for d in dataset]
            dataset = ret_dataset
        for d in dataset:
            d._dataset = path2dataset
        return np.array(dataset)
    # elif ext == ".db":
    #     return np.array(preprocess_database_data_multiprocessed(path2dataset, n_jobs=None))
    elif Path.is_dir(datasetPath):
        return mergeDatasets(loadDatasetsFromDirectory(datasetPath))
    raise IOError("Wrong file type.")


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
