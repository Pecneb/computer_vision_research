import json
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Dict, Any, Tuple, Optional
from multiprocessing import shared_memory
import gc

import joblib
import numpy as np
import tqdm
import h5py
import mpi4py.MPI as MPI

from .preprocessing import euclidean_distance
from dataManagementClasses import TrackedObject, Detection


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
    for p in tqdm.tqdm(dirPath.glob("*.joblib"), desc="Loading datasets"):
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
                load_dataset_with_labels,
                (p,),
                callback=loadDatasetMultiprocessedCallback,
            )
            dataset.append(tmpDatasetLabeled.get())
    return np.array(dataset)


def save_trajectories(
    trajectories: Union[List, np.ndarray],
    output: Union[str, Path],
    classifier: str = "SVM",
    name: str = "trajectories",
) -> List[str]:
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


def load_dataset(
    path2dataset: Union[str, List[str], Path], memmap_mode: Optional[str] = None
) -> np.ndarray:
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
        for p in tqdm.tqdm(path2dataset):
            datasets.append(load_dataset(p))
        return mergeDatasets(datasets)
    datasetPath = Path(path2dataset)
    ext = datasetPath.suffix
    if ext == ".joblib":
        try:
            dataset = joblib.load(path2dataset, memmap_mode)
        except Exception as e:
            print(f"Error loading {path2dataset}: {e}")
            return np.array([])
        if type(dataset[0]) == dict:
            ret_dataset = [d["track"] for d in dataset]
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


def load_shared_dataset(config: Dict[str, Any]) -> np.ndarray:
    """Load a shared dataset from a file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    np.ndarray
        Numpy array containing the dataset.

    """
    shm = shared_memory.SharedMemory(name=config["runtime"]["shm_name"])
    shape = tuple(config["runtime"]["dataset_shape"])
    dtype = np.dtype(config["runtime"]["dataset_dtype"])
    dataset = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
    shm.close()
    return dataset


def dataset_statistics(trackedObjects) -> Tuple[float, float, float, float, float]:
    """Calculate statistics of a dataset.

    Parameters
    ----------
    trackedObjects : list
        List of tracked objects.

    Returns
    -------
    Tuple[float, float, float, float, float]
        Tuple containing number of tracks, number of detections,
        average detections per track, max detections per track, min detections per track.
    """
    num_tracks = len(trackedObjects)
    num_detections = 0
    for o in trackedObjects:
        num_detections += len(o.history)
    avg_detections = num_detections / num_tracks
    max_detections = max([len(o.history) for o in trackedObjects])
    min_detections = min([len(o.history) for o in trackedObjects])
    std = np.std([len(o.history) for o in trackedObjects])
    distances = [
        euclidean_distance(
            p1=o.history_X[0],
            p2=o.history_Y[0],
            q1=o.history_X[-1],
            q2=o.history_Y[-1],
        )
        for o in trackedObjects
    ]
    max_distance = max(distances)
    min_distance = min(distances)
    return (
        num_tracks,
        num_detections,
        avg_detections,
        max_detections,
        min_detections,
        std,
        max_distance,
        min_distance,
    )


def joblib2h5py(joblib_file: Union[str, Path], h5py_file: Union[str, Path]):
    """Convert a joblib file to a h5py file.

    Parameters
    ----------
    joblib_file : Union[str, Path]
        Path to joblib file.
    h5py_file : Union[str, Path]
        Path to h5py file.

    """
    dataset: np.ndarray = load_dataset(joblib_file)
    with h5py.File(h5py_file, "w") as f:
        for i, tracked_object in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            grp = f.create_group(str(i))
            grp.create_dataset("id", data=tracked_object.objID)
            grp.create_dataset("history_X", data=tracked_object.history_X)
            grp.create_dataset("history_Y", data=tracked_object.history_Y)
            grp.create_dataset(
                "history_VX_calculated", data=tracked_object.history_VX_calculated
            )
            grp.create_dataset(
                "history_VY_calculated", data=tracked_object.history_VY_calculated
            )
            grp.create_dataset(
                "history_AX_calculated", data=tracked_object.history_AX_calculated
            )
            grp.create_dataset(
                "history_AY_calculated", data=tracked_object.history_AY_calculated
            )
            for j, detection in enumerate(tracked_object.history):
                dset = grp.create_group(f"detection_{j}")
                dset.create_dataset("label", data=detection.label)
                dset.create_dataset("confidence", data=detection.confidence)
                dset.create_dataset("frame_id", data=detection.frameID)
                dset.create_dataset("X", data=detection.X)
                dset.create_dataset("Y", data=detection.Y)
                dset.create_dataset("VX", data=detection.VX)
                dset.create_dataset("VY", data=detection.VY)
                dset.create_dataset("AX", data=detection.AX)
                dset.create_dataset("AY", data=detection.AY)
                dset.create_dataset("Width", data=detection.Width)
                dset.create_dataset("Height", data=detection.Height)


def load_dataset_from_h5py(
    h5py_file: Union[str, Path], verbose: bool = False
) -> np.ndarray:
    """Load a dataset from a h5py file.

    Parameters
    ----------
    h5py_file : Union[str, Path]
        Path to h5py file.

    Returns
    -------
    np.ndarray
        Numpy array containing the dataset.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset = []
    with h5py.File(h5py_file, "r", driver="mpio", comm=comm) as f:
        keys = list(f.keys())
        for i in tqdm.tqdm(keys[rank::size], total=len(keys) // size):
            grp = f[i]
            first_detection = grp[f"detection_0"]
            first_detection_obj = Detection(
                label=first_detection["label"][()],
                confidence=first_detection["confidence"][()],
                X=first_detection["X"][()],
                Y=first_detection["Y"][()],
                Width=first_detection["Width"][()],
                Height=first_detection["Height"][()],
                frameID=first_detection["frame_id"][()],
            )
            tracked_object = TrackedObject(
                id=grp["id"][()],
                first=first_detection_obj,
            )
            tracked_object.history_X = grp["history_X"][()]
            tracked_object.history_Y = grp["history_Y"][()]
            tracked_object.history_VX_calculated = grp["history_VX_calculated"][()]
            tracked_object.history_VY_calculated = grp["history_VY_calculated"][()]
            tracked_object.history_AX_calculated = grp["history_AX_calculated"][()]
            tracked_object.history_AY_calculated = grp["history_AY_calculated"][()]
            for j in grp.keys():
                if j.startswith("detection_"):
                    dset = grp[j]
                    detection = Detection(
                        label=dset["label"][()],
                        confidence=dset["confidence"][()],
                        X=dset["X"][()],
                        Y=dset["Y"][()],
                        Width=dset["Width"][()],
                        Height=dset["Height"][()],
                        frameID=dset["frame_id"][()],
                    )
                    detection.VX = dset["VX"][()]
                    detection.VY = dset["VY"][()]
                    detection.AX = dset["AX"][()]
                    detection.AY = dset["AY"][()]
                    tracked_object.history.append(detection)
            dataset.append(tracked_object)
            if verbose:
                print(
                    f"TrackedObject {tracked_object.objID} loaded with {len(tracked_object.history)} detections."
                )
    return np.array(dataset)


def object_to_dict(obj: TrackedObject) -> Dict[str, Any]:
    """Convert a TrackedObject to a dictionary.

    Parameters
    ----------
    obj : TrackedObject
        TrackedObject to convert.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the TrackedObject.
    """
    obj_dict = {
        "id": obj.objID,
        "history_X": obj.history_X.tolist(),
        "history_Y": obj.history_Y.tolist(),
        "history_VX_calculated": obj.history_VX_calculated.tolist(),
        "history_VY_calculated": obj.history_VY_calculated.tolist(),
        "history_AX_calculated": obj.history_AX_calculated.tolist(),
        "history_AY_calculated": obj.history_AY_calculated.tolist(),
        "detections": [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "X": detection.X,
                "Y": detection.Y,
                "Width": detection.Width,
                "Height": detection.Height,
                "frameID": detection.frameID,
                "VX": detection.VX,
                "VY": detection.VY,
                "AX": detection.AX,
                "AY": detection.AY,
            }
            for detection in obj.history
        ],
    }
    return obj_dict


def dataset_to_json(
    dataset: np.ndarray,
    json_file: Union[str, Path],
    verbose: bool = False,
    append: bool = False,
):
    """Convert a dataset to JSON format.

    Parameters
    ----------
    dataset : np.ndarray
        Numpy array containing the dataset.
    json_file : Union[str, Path]
        Path to the output JSON file.
    """
    json_data = []
    for tracked_object in tqdm.tqdm(dataset, desc="Converting to JSON"):
        obj_dict = object_to_dict(tracked_object)
        json_data.append(obj_dict)
        if verbose:
            print(f"TrackedObject {tracked_object.objID} converted to JSON.")
            print(f"History length: {len(tracked_object.history)}")
            # Check detection values
            for detection in zip(tracked_object.history, obj_dict["detections"]):
                assert detection[0].label == detection[1]["label"], "Label mismatch."
                assert (
                    detection[0].confidence == detection[1]["confidence"]
                ), "Confidence mismatch."
                assert detection[0].X == detection[1]["X"], "X mismatch."
                assert detection[0].Y == detection[1]["Y"], "Y mismatch."
                assert detection[0].Width == detection[1]["Width"], "Width mismatch."
                assert detection[0].Height == detection[1]["Height"], "Height mismatch."
                assert (
                    detection[0].frameID == detection[1]["frameID"]
                ), "FrameID mismatch."
                assert detection[0].VX == detection[1]["VX"], "VX mismatch."
                assert detection[0].VY == detection[1]["VY"], "VY mismatch."
                assert detection[0].AX == detection[1]["AX"], "AX mismatch."
                assert detection[0].AY == detection[1]["AY"], "AY mismatch."

    if append and Path(json_file).exists():
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                orig_json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading JSON file: {e}")
                orig_json_data = []
            except UnicodeDecodeError as e:
                print(f"Error loading JSON file: {e}")
                orig_json_data = []
        json_data = orig_json_data + json_data

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)


def append_to_json_file(data: List, json_file: Union[str, Path]):
    """Append data to a JSON file."""
    if Path(json_file).exists():
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                orig_json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading JSON file: {e}")
                orig_json_data = []
            except UnicodeDecodeError as e:
                print(f"Error loading JSON file: {e}")
                orig_json_data = []
        json_data = orig_json_data + data
    else:
        json_data = data

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)


def convert_joblib_to_json_chunked(
    joblib_files: List[Union[str, Path]],
    json_file: Union[str, Path],
    verbose: bool = False,
):
    """Convert multiple joblib files to a single JSON file."""
    for joblib_file in tqdm.tqdm(joblib_files, desc="Processing joblib files"):
        dataset = joblib.load(joblib_file)
        json_data = [object_to_dict(tracked_object) for tracked_object in dataset]
        append_to_json_file(json_data, json_file)
        if verbose:
            print(f"Appended data from {joblib_file} to {json_file}")
        # Free up memory
        del dataset
        del json_data
        gc.collect()
        


def load_dataset_from_json(json_file: Union[str, Path]) -> np.ndarray:
    """Load a dataset from a JSON file.

    Parameters
    ----------
    json_file : Union[str, Path]
        Path to the JSON file.

    Returns
    -------
    np.ndarray
        Numpy array containing the dataset.
    """
    with open(json_file, "rb") as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON file: {e}")
            json_data = json.loads(f.read())
        except UnicodeDecodeError as e:
            print(f"Error loading JSON file: {e}")
            json_data = json.loads(f.read())

    dataset = []
    for obj_dict in tqdm.tqdm(json_data, desc="Loading JSON"):
        first_detection_dict = obj_dict["detections"][0]
        first_detection = Detection(
            label=first_detection_dict["label"],
            confidence=first_detection_dict["confidence"],
            X=first_detection_dict["X"],
            Y=first_detection_dict["Y"],
            Width=first_detection_dict["Width"],
            Height=first_detection_dict["Height"],
            frameID=first_detection_dict["frameID"],
        )
        first_detection.VX = first_detection_dict["VX"]
        first_detection.VY = first_detection_dict["VY"]
        first_detection.AX = first_detection_dict["AX"]
        first_detection.AY = first_detection_dict["AY"]

        tracked_object = TrackedObject(
            id=obj_dict["id"],
            first=first_detection,
        )
        tracked_object.history_X = np.array(obj_dict["history_X"])
        tracked_object.history_Y = np.array(obj_dict["history_Y"])
        tracked_object.history_VX_calculated = np.array(
            obj_dict["history_VX_calculated"]
        )
        tracked_object.history_VY_calculated = np.array(
            obj_dict["history_VY_calculated"]
        )
        tracked_object.history_AX_calculated = np.array(
            obj_dict["history_AX_calculated"]
        )
        tracked_object.history_AY_calculated = np.array(
            obj_dict["history_AY_calculated"]
        )

        tmp_detections = []
        for detection_dict in obj_dict["detections"]:
            detection = Detection(
                label=detection_dict["label"],
                confidence=detection_dict["confidence"],
                X=detection_dict["X"],
                Y=detection_dict["Y"],
                Width=detection_dict["Width"],
                Height=detection_dict["Height"],
                frameID=detection_dict["frameID"],
            )
            detection.VX = detection_dict["VX"]
            detection.VY = detection_dict["VY"]
            detection.AX = detection_dict["AX"]
            detection.AY = detection_dict["AY"]
            tmp_detections.append(detection)
        tracked_object.history = tmp_detections

        dataset.append(tracked_object)

    return np.array(dataset)
