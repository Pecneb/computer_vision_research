import logging
import numpy as np
import tqdm

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

def filter_by_class(trackedObjects: list, label="car"):
    """Only return objects with given label.

    Args:
        trackedObjects (list): list of tracked objects 
        label (str, optional): label of object. Defaults to "car".

    Returns:
        list: list of "label" objects 
    """
    return [obj for obj in trackedObjects if obj.label==label]

def shuffle_data(trackedObjects: list) -> list:
    rng = np.random.default_rng()
    for i in range(len(trackedObjects)):
        randIDX = int(len(trackedObjects) * rng.random())
        tmpObj = trackedObjects[i]
        trackedObjects[i] = trackedObjects[randIDX]
        trackedObjects[randIDX] = tmpObj
    
