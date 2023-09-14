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
from datetime import date 
from pathlib import Path
from typing import List

### Third Party ###
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import icecream
from tqdm import tqdm
from icecream import ic
icecream.install()

np.seterr(divide='ignore', invalid='ignore')

### Local ###
from utility.models import (
   load_model,
   save_model 
)
from utility.dataset import (
    load_dataset,
    save_trajectories,
    preprocess_database_data_multiprocessed
)
from utility.general import (
    strfy_dict_params
)
from utility.preprocessing import (
    filter_trajectories,
    filter_by_class,
)
from utility.training import (
    iter_minibatches
)
from clustering import make_4D_feature_vectors, make_6D_feature_vectors, calc_cluster_centers
from dataManagementClasses import insert_weights_into_feature_vector

import numpy as np
import tqdm


def make_features_for_classification(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_features_for_classification_velocity(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_feature_vectors_version_one(trackedObjects: List, k: int, labels: np.ndarray = None, reduced_labels: np.ndarray = None, up_until: float = 1):
    """Make feature vectors for classification algorithm

    Args:
        trackedObjects (list): Tracked objects 
        k (int): K is the number of slices, the object history should be sliced up into.
        labels (np.ndarray): Results of clustering.

    Returns:
        Tuple[ndarray, ndarray, ndarray, ndarray]: Tuple of arrays. First array is the feature vector array,
            second array is the labels array, third is the metadat array, and the last is the reduced labels
            array.
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
                if labels is not None:
                    newLabels.append(labels[j])
                if reduced_labels is not None:
                    newReducedLabels.append(reduced_labels[j])
                track_history_metadata.append([trackedObjects[j].history[i].frameID, trackedObjects[j].history[i+midstep].frameID, 
                trackedObjects[j].history[i+step].frameID, len(trackedObjects[j].history), trackedObjects[j]])
    return np.array(featureVectors), np.array(newLabels), np.array(track_history_metadata), np.array(newReducedLabels)

def make_feature_vectors_version_one_half(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_feature_vectors_version_two(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_feature_vectors_version_two_half(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_feature_vectors_version_three(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_feature_vectors_version_three_half(trackedObjects: List, k: int, labels: np.ndarray):
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

def make_feature_vectors_version_four(trackedObjects: List, max_stride: int, labels: np.ndarray):
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

def make_feature_vectors_version_five(trackedObjects: List, labels: np.ndarray, max_stride: int, n_weights: int):
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

def make_feature_vectors_version_six(trackedObjects: List, labels: np.ndarray, max_stride: int, weights: np.ndarray):
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

def make_feature_vectors_version_seven(trackedObjects: List, labels: np.ndarray, max_stride: int):
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

def KNNClassification(X: np.ndarray, y: np.ndarray, n_neighbours: int):
    """Run K Nearest Neighbours classification on samples X and labels y with neighbour numbers n_neighbours.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels
        n_neighbours (int): Number of neighbours to belong in a class.

    Returns:
        sklearn classifier: KNN model
    """
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbours, weights='distance').fit(X, y)
    return classifier

def SGDClassification(X: np.ndarray, y: np.ndarray):
    """Run Stochastic Gradient Descent Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: SGD model 
    """
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(loss="modified_huber").fit(X, y)
    return classifier

def GPClassification(X: np.ndarray, y: np.ndarray):
    """Run Gaussian Process Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: GP model 
    """
    from sklearn.gaussian_process import GaussianProcessClassifier
    classifier = GaussianProcessClassifier().fit(X, y)
    return classifier

def GNBClassification(X: np.ndarray, y: np.ndarray):
    """Run Gaussian Naive Bayes Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: GNB model 
    """
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB().fit(X, y)
    return classifier

def MLPClassification(X: np.ndarray, y: np.ndarray):
    """Run Multi Layer Perceptron Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: MLPC model 
    """
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(max_iter=1000).fit(X,y)
    return classifier

def VotingClassification(X: np.ndarray, y: np.ndarray):
    """Run Voting Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: Voting model 
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    clf1 = KNeighborsClassifier(n_neighbors=15, weights='distance')
    clf2 = SGDClassifier()
    clf3 = GaussianProcessClassifier()
    clf4 = GaussianNB()
    clf5 = MLPClassifier()
    classifier = VotingClassifier(
        estimators=[('knn', clf1), ('sgd', clf2), ('gp', clf3), ('gnb', clf4), ('mlp', clf5)]
    ).fit(X, y)
    return classifier

def SVMClassficitaion(X: np.ndarray, y: np.ndarray):
    """Run Support Vector Machine classification with RBF kernel.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: SVM model
    """
    from sklearn.svm import SVC
    classifier = SVC().fit(X, y)
    return classifier

def DTClassification(X: np.ndarray, y: np.ndarray):
    """Run decision tree classification.

    Args:
        X (np.ndarray): Dataset 
        y (np.ndarray): labels
    """
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier().fit(X, y)
    return classifier

def Classification(classifier: str, path2db: str, **argv):
    """Run classification on database data.

    Args:
        classifier (str): Type of the classifier. 
        path2db (str): Path to database file. 

    Returns:
        bool: Returns false if bad classifier was given. 
    """
    X_train, y_train, _, X_valid, y_valid, _, _ = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    fig, ax = plt.subplots()
    model = None
    if classifier == 'KNN':
        model = KNNClassification(X_train, y_train, 15)
    elif classifier == 'SGD':
        model = SGDClassification(X_train, y_train)
    elif classifier == 'GP':
        model = GPClassification(X_train, y_train)
    elif classifier == 'GNB':
        model = GNBClassification(X_train, y_train)
    elif classifier == 'MLP':
        model = MLPClassification(X_train, y_train)
    elif classifier == 'VOTE':
        model = VotingClassification(X_train, y_train)
    elif classifier == 'SVM':
        model = SVMClassficitaion(X_train, y_train)
    elif classifier == 'DT':
        model = DTClassification(X_train, y_train)
    else:
        print(f"Error: bad classifier {classifier}")
        return False
    ValidateClassification(model, X_valid, y_valid)
    """xx, yy= np.meshgrid(np.arange(0, 2, 0.005), np.arange(0, 2, 0.005))
    X_visualize = np.zeros(shape=(xx.shape[0]*xx.shape[1],6))
    counter = 0
    for i in range(0,xx.shape[0]):
        for j in range(0,xx.shape[1]):
            X_visualize[counter,0] = xx[j,i]
            X_visualize[counter,1] = yy[j,i]
            X_visualize[counter,2] = xx[j,i]
            X_visualize[counter,3] = yy[j,i]
            X_visualize[counter,4] = xx[j,i]
            X_visualize[counter,5] = yy[j,i]
            counter += 1
    y_visualize = model.predict(X_visualize)
    ax.pcolormesh(xx,yy,y_visualize.reshape(xx.shape))
    ax.scatter(X_train[:, 0], 1-X_train[:, 1], c=y_train, edgecolors='k')
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    plt.show()"""
    save_model(path2db, str("model_"+classifier), model)

def ClassificationWorker(path2db: str, **argv):
    """Run all of the classification methods implemented.

    Args:
        path2db (str): Path to database file. 
    """
    X_train, y_train, _, X_valid, y_valid, _, _= data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    print("KNN")
    model = KNNClassification(X_train, y_train, 15)
    ValidateClassification(model, X_valid, y_valid)
    print("SGD")
    model = SGDClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid) 
    print("GP")
    model = GPClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("GNB")
    model = GNBClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("MLP")
    model = MLPClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("VOTE")
    model = VotingClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("SVM")
    model = SVMClassficitaion(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("DT")
    model = DTClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)

def ValidateClassification(clfmodel, X_valid: np.ndarray, y_valid: np.ndarray):
    """Validate fitted classification model.

    Args:
        clfmodel (str, model): Can be a path to model file or model. 
        X_valid (np.ndarray): Test dataset. 
        y_valid (np.ndarray): Test dataset's labeling. 
    """
    if type(clfmodel) is str:
        model = joblib.load(clfmodel)
    else:
        model = clfmodel
    y_predict = model.predict(X_valid)
    assert len(y_predict) == len(y_valid)
    print(f"Number of mislabeled points out of a total {X_valid.shape[0]} points : {(y_valid != y_predict).sum()} \nAccuracy: {(1-((y_valid != y_predict).sum() / X_valid.shape[0])) * 100} %")

def ValidateClassification_Probability(clfmodel, X_valid: np.ndarray, y_valid: np.ndarray, threshold: np.float64):
    """Calculate accuracy of classification model using the predict_proba method of the classifier.

    Args:
        clfmodel (str, model): Can be a path to model file or model. 
        X_valid (np.ndarray): Test dataset. 
        y_valid (np.ndarray):  Test dataset's labeling.
    """
    if type(clfmodel) is str:
        model = joblib.load(clfmodel)
    else:
        model = clfmodel
    y_predict_proba = model.predict_proba(X_valid)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, y_proba_vec in enumerate(y_predict_proba):
        for j, y_proba in enumerate(y_proba_vec):
            if j == y_valid[i]:
                if y_proba >= threshold:
                    tp += 1 
                else:
                    fn += 1
            else:
                if y_proba < threshold:
                    tn += 1
                else:
                    fp += 1
    return True
                
def CalibratedClassification(classifier: str, path2db: str, **argv):
    """Run classification on database data.

    Args:
        classifier (str): Type of the classifier. 
        path2db (str): Path to database file. 

    Returns:
        bool: Returns false if bad classifier was given. 
    """
    from sklearn.calibration import CalibratedClassifierCV
    X_train, y_train, X_calib, y_calib, X_valid, y_valid = data_preprocessing_for_calibrated_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    fig, ax = plt.subplots()
    model = None
    if classifier == 'KNN':
        model = KNNClassification(X_train, y_train, 15)
    elif classifier == 'SGD':
        model = SGDClassification(X_train, y_train)
    elif classifier == 'GP':
        model = GPClassification(X_train, y_train)
    elif classifier == 'GNB':
        model = GNBClassification(X_train, y_train)
    elif classifier == 'MLP':
        model = MLPClassification(X_train, y_train)
    elif classifier == 'VOTE':
        model = VotingClassification(X_train, y_train)
    else:
        print(f"Error: bad classifier {classifier}")
        return False
    model_calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_calibrated.fit(X_calib, y_calib)
    ValidateClassification(model_calibrated, X_valid, y_valid)
    xx, yy= np.meshgrid(np.arange(0, 2, 0.005), np.arange(0, 2, 0.005))
    X_visualize = np.zeros(shape=(xx.shape[0]*xx.shape[1],6))
    counter = 0
    for i in range(0,xx.shape[0]):
        for j in range(0,xx.shape[1]):
            X_visualize[counter,0] = xx[j,i]
            X_visualize[counter,1] = yy[j,i]
            X_visualize[counter,2] = xx[j,i]
            X_visualize[counter,3] = yy[j,i]
            X_visualize[counter,4] = xx[j,i]
            X_visualize[counter,5] = yy[j,i]
            counter += 1
    y_visualize = model.predict(X_visualize)
    ax.pcolormesh(xx,yy,y_visualize.reshape(xx.shape))
    ax.scatter(X_train[:, 0], 1-X_train[:, 1], c=y_train, edgecolors='k')
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    plt.show()
    save_model(path2db, str("calibrated_model_"+classifier), model_calibrated)

def CalibratedClassificationWorker(path2db: str, **argv):
    """Run all the classification methods implemented.

    Args:
        path2db (str): Path to database file. 
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    X_train, y_train, _, X_valid, y_valid, _, _ = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    #vote = VotingClassification(X_train, y_train)
    models = {
        'KNN' : KNeighborsClassifier(n_neighbors=15),
        'SGD' : SGDClassifier(),
        'GP' : GaussianProcessClassifier(n_jobs=argv['n_jobs']),
        'GNB' : GaussianNB(),
        'MLP' : MLPClassifier()
    }
    for cls in models:
        print(cls)
        calibrated = CalibratedClassifierCV(models[cls], method="sigmoid", n_jobs=18).fit(X_train, y_train)
        ValidateClassification(calibrated, X_valid, y_valid)

def BinaryClassificationWorkerTrain(path2db: str, path2model = None, **argv):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from classifier import OneVSRestClassifierExtended
    from sklearn.tree import DecisionTreeClassifier

    X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, tracks = [], [], [], [], [], [], []

    if path2model is not None:
        model = load_model(path2model)
        tracks = model.tracks
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid = data_preprocessing_for_classifier_from_joblib_model(model, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'], 
                                                            from_half=argv['from_half'],
                                                            features_v2=argv['features_v2'],
                                                            features_v2_half=argv['features_v2_half'],
                                                            features_v3=argv['features_v3'])
    else:
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, tracks = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'], from_half=argv['from_half'],
                                                            features_v2=argv['features_v2'],
                                                            features_v2_half=argv['features_v2_half'],
                                                            features_v3=argv['features_v3'])

    """X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, tracks, cluster_centroids = preprocess_dataset_for_training(
        path2dataset=path2db, 
        min_samples=argv['min_samples'], 
        max_eps=argv['max_eps'], 
        xi=argv['xi'], 
        min_cluster_size=argv['min_cluster_size'],
        n_jobs=argv['n_jobs'], 
        from_half=argv['from_half'],
        features_v2=argv['features_v2'],
        features_v2_half=argv['features_v2_half'],
        features_v3=argv['features_v3']
    )"""

    models = {
        'KNN' : KNeighborsClassifier,
        'GP' : GaussianProcessClassifier,
        'GNB' : GaussianNB,
        'MLP' : MLPClassifier,
        'SGD' : SGDClassifier,
        'SVM' : SVC,
        'DT' : DecisionTreeClassifier
    }
    
    parameters = {
        'KNN' : {'n_neighbors' : 15},
        'GP' :  {},
        'GNB' : {},
        'MLP' : {'max_iter' : 1000, 'solver' : 'sgd'},
        'SGD' : {'loss' : 'modified_huber'},
        'SVM' : {'kernel' : 'rbf', 'probability' : True},
        'DT' : {} 
    }

    table = pd.DataFrame()
    table2 = pd.DataFrame()
    probability_over_time = pd.DataFrame()

    if not os.path.isdir(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables")):
            os.mkdir(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables"))
    savepath = os.path.join(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables"))

    for clr in tqdm(models, desc="Classifier trained."):
        binaryModel = OneVSRestClassifierExtended(models[clr](**parameters[clr]), tracks, n_jobs=argv['n_jobs'])
        #binaryModel = BinaryClassifier(trackData=tracks, classifier=models[clr], classifier_argv=parameters[clr])
        #binaryModel.init_models(models[clr])

        binaryModel.fit(X_train, y_train)

        top_picks = []
        for i in range(1,4):
            top_picks.append(binaryModel.validate_predictions(X_valid, y_valid, top=i))
        balanced_threshold = binaryModel.validate(X_valid, y_valid, argv['threshold'])

        table[clr] = np.asarray(top_picks)
        table2[clr] = balanced_threshold

        probabilities = binaryModel.predict_proba(X_valid)
        for i in range(probabilities.shape[1]):
            probability_over_time[f"Class {i}"] = probabilities[:, i]
        probability_over_time["Time_Enter"] = metadata_valid[:, 0]
        probability_over_time["Time_Mid"] = metadata_valid[:, 1]
        probability_over_time["Time_Exit"] = metadata_valid[:, 2]
        probability_over_time["History_Length"] = metadata_valid[:, 3]
        probability_over_time["TrackID"] = metadata_valid[:, 4]
        probability_over_time["True_Class"] = y_valid 

        filename = os.path.join(savepath, f"{date.today()}_{clr}.xlsx")
        with pd.ExcelWriter(filename) as writer:
            probability_over_time.to_excel(writer, sheet_name="Probability_over_time") # each feature vector
            table.to_excel(writer, sheet_name="Top_Picks") # top n accuracy
            table2.to_excel(writer, sheet_name="Balanced") # balanced accuracy

        #TODO: somehow show in title which feature vectors were used for the tarining 
        if argv['from_half']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_from_half"), binaryModel) 
        elif argv['features_v2']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2"), binaryModel)
        elif argv['features_v2_half']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2_from_half"), binaryModel)
        elif argv['features_v3']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v3"), binaryModel)


    table.index += 1
    print("Top picks")
    print(table.to_markdown())
    print("Threshold")
    print(table2.to_markdown())
    print(table2.aggregate(np.average).to_markdown())

def train_binary_classifiers(path2dataset: str, outdir: str, **argv):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from classifier import OneVSRestClassifierExtended
    from sklearn.tree import DecisionTreeClassifier
    from visualizer import aoiextraction

    X, y, metadata, tracks, labels = preprocess_dataset_for_training(
        path2dataset=path2dataset, 
        min_samples=argv['min_samples'], 
        max_eps=argv['max_eps'], 
        xi=argv['xi'], 
        min_cluster_size=argv['min_cluster_size'],
        n_jobs=argv['n_jobs'], 
        cluster_features_version=argv['cluster_features_version'],
        classification_features_version=argv['classification_features_version'],
        stride=argv['stride'],
        level=argv['level'],
        threshold=argv['threshold'],
        n_weights=argv['n_weights'],
        weights_preset=argv['weights_preset'],
        p_norm=argv['p_norm']
    )
    tracks_filtered = [t for i, t in enumerate(tracks) if labels[i] > -1] 
    labels_filtered = [l for l in labels if l > -1]  

    tracks_labels = []
    for i in range(len(tracks_filtered)):
        tracks_labels.append({
            "track" : tracks_filtered[i],
            "class" : labels_filtered[i]
        })

    cluster_centroids = None
    if argv['classification_features_version'] == 'v3' or argv['classification_features_version'] == 'v3_half':
        cluster_centroids = aoiextraction(tracks_filtered, labels_filtered)

    models = {
        'KNN' : KNeighborsClassifier,
        #'GP' : GaussianProcessClassifier,
        'SVM' : SVC,
        'DT' : DecisionTreeClassifier
    }
    
    parameters = {
        'KNN' : {'n_neighbors' : 15},
        #'GP' :  {},
        'SVM' : {'kernel' : 'rbf', 'probability' : True, 'max_iter' : 26000},
        'DT': {}
    }

    if not os.path.isdir(os.path.join(outdir, "tables")):
            os.mkdir(os.path.join(outdir, "tables"))

    if argv['batch_size'] is not None:
        batch_size = argv['batch_size']
        if X.shape[0] < batch_size:
            batch_size = X.shape[0]

    all_classes = np.array(list(set(y)))

    for clr in tqdm(models, desc="Classifier trained."):
        binaryModel = OneVSRestClassifierExtended(models[clr](**parameters[clr]), tracks_labels, n_jobs=argv['n_jobs'])

        # if batch size is given, use partial_fit() method and train with minibatches
        if argv['batch_size'] is not None:
            try:
                iteration = 1
                for X_batch, y_batch in iter_minibatches(X, y, batch_size):
                    print(f"Iteration {iteration} started")
                    binaryModel.partial_fit(X_batch, y_batch, classes=all_classes, centroids=cluster_centroids)
                    iteration+=1
                print(f"\nTraining with batchsize: {batch_size:10d}.\n")
            except:
                print(f"\nClassifier {clr} does not have partial_fit() method, cant train with minibatches.")
                print(f"Training without minibatches. Batchsize is: {X.shape[0]:10d}\n")
                binaryModel.fit(X, y, centroids=cluster_centroids)            
        else:
            print(f"\nTraining without minibatches. Batchsize is: {X.shape[0]:10d}\n")
            binaryModel.fit(X, y, centroids=cluster_centroids)

        # save models with names corresponding to the feature version and parameters
        if argv['classification_features_version'] == 'v1':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])), binaryModel)
        elif argv['classification_features_version'] == 'v1_half':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_from_half"), binaryModel) 
        elif argv['classification_features_version'] == 'v2':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2"), binaryModel)
        elif argv['classification_features_version'] == 'v2_half':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2_from_half"), binaryModel)
        elif argv['classification_features_version'] == 'v3':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v3"), binaryModel)
        elif argv['classification_features_version'] == 'v3_half':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v3_from_half"), binaryModel)
        elif argv['classification_features_version'] == 'v4':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v4"), binaryModel)
        elif argv['classification_features_version'] == 'v5':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+f"_{argv['n_weights']}_v5"), binaryModel)
        elif argv['classification_features_version'] == 'v6':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+f"_{argv['n_weights']}_v6"), binaryModel)
        elif argv['classification_features_version'] == 'v7':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+f"_stride-{argv['stride']}_v7"), binaryModel)
    
def BinaryClassificationTrain(classifier: str, path2db: str, **argv):
    """Deprecated, dont use.

    Will update in time.

    Args:
        classifier (str): _description_
        path2db (str): _description_
    """
    print("Warning: deprecated function, dont use.")
    print("Exiting...")
    exit(1)
    from classifier import BinaryClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    X_train, y_train, X_valid, y_valid, tracks = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    table = pd.DataFrame()
    binaryModel = BinaryClassifier(X_train, y_train, tracks)
    if classifier == 'KNN':
        binaryModel.init_models(KNeighborsClassifier, n_neighbors=15)
    if classifier == 'MLP':
        binaryModel.init_models(MLPClassifier, max_iter=1000, solver="sgd")
    if classifier == 'SGD':
        binaryModel.init_models(SGDClassifier, loss="modified_huber")
    if classifier == 'GP':
        binaryModel.init_models(GaussianProcessClassifier)
    if classifier == 'GNB':
        binaryModel.init_models(GaussianNB)
    if classifier == 'SVM':
        binaryModel.init_models(SVC, kernel='rbf', probability=True)
    binaryModel.fit()
    accuracy_vector = binaryModel.validate(X_valid, y_valid, 0.8)
    table[classifier] = accuracy_vector # add col to pandas dataframe
    save_model(path2db, str("binary_"+classifier), binaryModel) 
    print(table.to_markdown()) # print out pandas dataframe in markdown table format.

def BinaryDecisionTreeClassification(path2dataset: str, min_samples: int, max_eps: float, xi: float, min_cluster_size: int, n_jobs: int, from_half=False):
    from classifier import BinaryClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid = [], [], [], [] , [], [] 

    trackData = []

    threshold = 0.5
    
    if path2dataset.split(".")[-1] == "db":
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, trackData = data_preprocessing_for_classifier(
            path2dataset, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, from_half=from_half)
    elif path2dataset.split(".")[-1] == "joblib":
        model = load_model(path2dataset)
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid = data_preprocessing_for_classifier_from_joblib_model(
            model=model, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs, from_half=from_half)
        trackData = model.trackData

    for d in range(2, 11):
        table_one = pd.DataFrame()
        # Initialize BinaryClassifier
        binaryModel = BinaryClassifier(X_train, y_train, trackData)
        binaryModel.init_models(DecisionTreeClassifier, max_depth=d)
        binaryModel.fit()
        # Validate BinaryClassifier
        predict_proba_balanced_accuracy = binaryModel.validate(X_valid, y_valid, threshold=threshold) # Validating the predict_proba() mathod, that returns back probability for every class
        predict_accuracy = binaryModel.validate_predictions(X_valid, y_valid, threshold=threshold) # validating predict() method, that returns only the highest predicted class
        # Create tables for accurcy
        table_one[f"Depth {d}"] = predict_proba_balanced_accuracy
        table_one.loc[0, f"Depth {d} multiclass average"] = np.average(predict_proba_balanced_accuracy)
        table_one.loc[0, f"Depth {d} one class prediction"] = predict_accuracy 
        # print out table in markdown
        print(f"Decision Tree depth {d} accuracy")
        print(table_one.to_markdown())

def validate_models(path2models: str, **argv):
    """Validate trained classifiers.

    Args:
        path2models (str): Path to parent directory containing models.
    """
    import datetime
    filenames = os.listdir(path2models)
    models = []
    classifier_names = []
    for n in filenames:
        if n.startswith("binary") and n.endswith(".joblib"):
            models.append(load_model(os.path.join(path2models, n)))
            classifier_names.append(n.split("_")[1].split(".")[0])

    table = pd.DataFrame()
    table2 = pd.DataFrame()
    probability_over_time = pd.DataFrame()

    
    if not os.path.isdir(os.path.join(*path2models.split("/")[:-1], "tables")):
        os.mkdir(os.path.join(*path2models.split("/")[:-1], "tables"))
    savepath = os.path.join(os.path.join(*path2models.split("/")[:-1], "tables"))

    _, _, _, X_valid, y_valid, metadata_valid = data_preprocessing_for_classifier_from_joblib_model(
        models[1], min_samples=argv["min_samples"], max_eps=argv["max_eps"], xi=argv["xi"],
        min_cluster_size=argv["min_cluster_size"], n_jobs=argv["n_jobs"],
        features_v2=argv['features_v2'], features_v2_half=argv['features_v2_half'])

    for clr, m in zip(classifier_names, models):
        top_picks = []
        for i in range(1,4):
            top_picks.append(m.validate_predictions(X_valid, y_valid, top=i))
        balanced_threshold = m.validate(X_valid, y_valid, argv['threshold'])
        # print(np.asarray(top_picks) )
        table[clr] = np.asarray(top_picks)
        table2[clr] = balanced_threshold

        probabilities = m.predict_proba(X_valid)
        for i in range(probabilities.shape[1]):
            probability_over_time[f"Class {i}"] = probabilities[:, i]
        probability_over_time["Time_Enter"] = metadata_valid[:, 0]
        probability_over_time["Time_Mid"] = metadata_valid[:, 1]
        probability_over_time["Time_Exit"] = metadata_valid[:, 2]
        probability_over_time["History_Length"] = metadata_valid[:, 3]
        probability_over_time["TrackID"] = metadata_valid[:, 4]
        probability_over_time["True_Class"] = y_valid  

        filename = os.path.join(savepath, f"{datetime.date.today()}_{clr}.xlsx")
        with pd.ExcelWriter(filename) as writer:
            probability_over_time.to_excel(writer, sheet_name="Probability_over_time")

    print("Top picks")
    print(table.to_markdown())
    print("Threshold")
    print(table2.to_markdown())
    print(table2.aggregate(np.average).to_markdown())

def true_class_under_threshold(predictions: np.ndarray, true_classes: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """Return numpy array of featurevectors that's predictions for their true class is under given threshold.

    Args:
        predictions (np.ndarray): Probability vectors. 
        true_classes (np.ndarray): Numpy array of the true classes ordered to feature vectors.
        X (np.ndarray): Feature vectors. 
        threshold (float): Threshold.

    Returns:
        np.ndarray: numpy array of feature vectors, that's true class's prediction probability is under threshold.
    """
    return_vector = []
    for i, pred in enumerate(predictions):
        if pred[true_classes[i]] < threshold:
            return_vector.append(X[i])
    return np.array(return_vector)

def all_class_under_threshold(predictions: np.ndarray, true_classes: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """Return numpy array of features that's predictions for all classes are under the given threshold.

    Args:
        predictions (np.ndarray): Probability vectors. 
        true_classes (np.ndarray): Numpy array of the true classes ordered to feature vectors.
        X (np.ndarray): Feature vectors. 
        threshold (float): Threshold.

    Returns:
        np.ndarray: numpy array of feature vectors, that's classes prediction probability is under threshold.
    """
    return_vector = []
    for i, preds in enumerate(predictions):
        renitent = True
        for pred in preds:
            if pred > threshold:
                renitent = False 
        if renitent:
            return_vector.append(X[i])
    return np.array(return_vector)

def investigateRenitent(path2model: str, threshold: float, **argv):
    """Filter out renitent predictions, that cant predict which class the detections is really in.

    Args:
        path2model (str): Path to model. 
    """
    model = load_model(path2model)
    _, _, _, X_test, y_test, _ = data_preprocessing_for_classifier_from_joblib_model(
        model, min_samples=argv["min_samples"], max_eps=argv["max_eps"], xi=argv["xi"],
        min_cluster_size=argv["min_cluster_size"], n_jobs=argv["n_jobs"])

    probas = model.predict_proba(X_test)

    renitent_vector = true_class_under_threshold(probas, y_test, X_test, threshold)

    renitent_vector_2 = all_class_under_threshold(probas, y_test, X_test, threshold)

    fig, ax = plt.subplots(1, 2)

    if len(renitent_vector) > 0:
        ax[0].set_title(f"Renitent: true class under threshold {threshold}: {len(renitent_vector)}")
        ax[0].scatter(renitent_vector[:, 0], 1 - renitent_vector[:, 1], s=2.5, c='g')
        ax[0].scatter(renitent_vector[:, 4], 1 - renitent_vector[:, 5], s=2.5)
        ax[0].scatter(renitent_vector[:, 6], 1 - renitent_vector[:, 7], s=2.5, c='r')
        print(f"There are {len(renitent_vector)} renitent detections out of {len(X_test)}.")
    else:
        print(f"Renitent: true class under threshold {threshold}")

    if len(renitent_vector_2) > 0:
        ax[1].set_title(f"Renitent: classes under threshold {threshold}: {len(renitent_vector_2)}")
        ax[1].scatter(renitent_vector_2[:, 0], 1 - renitent_vector_2[:, 1], s=2.5, c='g')
        ax[1].scatter(renitent_vector_2[:, 4], 1 - renitent_vector_2[:, 5], s=2.5)
        ax[1].scatter(renitent_vector_2[:, 6], 1 - renitent_vector_2[:, 7], s=2.5, c='r')
        print(f"There are {len(renitent_vector_2)} renitent detections out of {len(X_test)}.")
    else:
        print(f"Renitent: classes under threshold {threshold}")
        
    plt.show()

def plot_decision_tree(path2model: str):
    """Draw out the decision tree in a tree graph.

    Args:
        path2model (str): Path to the joblib binary model file.
    """
    from sklearn.tree import plot_tree
    model = load_model(path2model=path2model)
    for i, m in enumerate(model.estimators_):
        print(f"Class {i}")
        plot_tree(m)
        plt.show()

def cross_validate(path2dataset: str, outputPath: str = None, train_ratio=0.75, seed=1, n_splits=5, n_jobs=18, estimator_params_set=1, classification_features_version: str = "v1", stride: int = 15, level: float = None, n_weights: int = 3, weights_preset: int = 1, threshold: float = 0.7, **estkwargs):
    """Run cross validation on chosen classifiers with different feature vectors.

    Args:
        path2dataset (str): dataset path.
        outputPath (str, optional): The path to the output table to save the results. Defaults to None.
        train_ratio (float, optional): The ratio of the training dataset compared to the test dataset. Defaults to 0.75.
        seed (int, optional): Seed value to be able reproduce shuffle on dataset. Defaults to 1.
        n_splits (int, optional): Cross validation split number. Defaults to 5.
        n_jobs (int, optional): Number of paralell processes to run. Defaults to 18.
        estimator_params_set (int, optional): Choose classifier parameter set. Defaults to 1.
        classification_features_version (str, optional): Choose which feature vector to use for classification. Defaults to "v1".
        stride (int, optional): Size of the sliding window used to generate feature vectors from detection history. Defaults to 15.
        level (bool, optional): Choose if dataset should be balanced or stay as it is after enrichment. Defaults to False.
        n_weights (int, optional): The number of dimensions, that is going to be added between the first and last dimension in the feature vector. Defaults to n_weights.

    Returns:
        tuple: cross validation results in pandas datastructure 
    """
    from clustering import clustering_on_feature_vectors
    from sklearn.cluster import OPTICS
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from classifier import OneVSRestClassifierExtended
    from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
    from sklearn.metrics import top_k_accuracy_score, make_scorer, balanced_accuracy_score
    from visualizer import aoiextraction

    tracks = load_dataset(path2dataset)

    # cluster tracks
    tracks_filtered = filter_trajectories(trackedObjects=tracks, threshold=threshold)
    cls_samples = make_4D_feature_vectors(tracks_filtered)
    _, labels = clustering_on_feature_vectors(X=cls_samples, estimator=OPTICS, n_jobs=n_jobs, **estkwargs)
    tracks_labeled = tracks_filtered[labels > -1]
    cluster_labels = labels[labels > -1]

    train_dict = []
    for i in range(len(tracks_labeled)):
        train_dict.append({
            "track" : tracks_labeled[i],
            "class" : cluster_labels[i]
        })

    # shuffle tracks, and separate into a train and test dataset
    train, test = train_test_split(train_dict, train_size=train_ratio, random_state=seed)

    tracks_train = [t["track"] for t in train]
    labels_train = np.array([t["class"] for t in train])
    tracks_test = [t["track"] for t in test]
    labels_test = np.array([t["class"] for t in test])

    cluster_centroids = None
    fit_params = None

    if classification_features_version == "v1":
        X_train, y_train, metadata_train = make_feature_vectors_version_one(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_one(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v1_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_one_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_one_half(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v2":
        X_train, y_train, metadata_train = make_feature_vectors_version_two(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_two(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v2_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_two_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_two_half(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v3":
        X_train, y_train, metadata_train = make_feature_vectors_version_three(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_three(trackedObjects=tracks_test, k=6, labels=labels_test)
        cluster_centroids = aoiextraction([t["track"] for t in tracks_filtered], [t["class"] for t in tracks_filtered]) 
        fit_params = {
            'centroids' : cluster_centroids
        }
    elif classification_features_version == "v3_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_three_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_three_half(trackedObjects=tracks_test, k=6, labels=labels_test)
        cluster_centroids = aoiextraction([t["track"] for t in tracks_filtered], [t["class"] for t in tracks_filtered]) 
        fit_params = {
            'centroids' : cluster_centroids
        }
    elif classification_features_version == "v4":
        X_train, y_train, metadata_train = make_feature_vectors_version_four(trackedObjects=tracks_train, max_stride=stride, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_four(trackedObjects=tracks_test, max_stride=stride, labels=labels_test)
    elif classification_features_version == "v5":
        X_train, y_train, metadata_train = make_feature_vectors_version_five(trackedObjects=tracks_train, labels=labels_train, max_stride=stride, n_weights=n_weights)
        X_test, y_test, metadata_test = make_feature_vectors_version_five(trackedObjects=tracks_test, labels=labels_test, max_stride=stride, n_weights=n_weights)
    elif classification_features_version == "v6":
        weights_presets = {
            1 : np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0]),
            2 : np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        }
        X_train, y_train, metadata_train = make_feature_vectors_version_six(trackedObjects=tracks_train, labels=labels_train, max_stride=stride, weights=weights_presets[weights_preset])
        X_test, y_test, metadata_test = make_feature_vectors_version_six(trackedObjects=tracks_test, labels=labels_test, max_stride=stride, weights=weights_presets[weights_preset])
    if level is not None:
        X_train, y_train = level_features(X_train, y_train, level)
        X_test, y_test = level_features(X_test, y_test, level)
    elif classification_features_version == "v7":
        X_train, y_train, _ = make_feature_vectors_version_seven(trackedObjects=tracks_train, labels=labels_train, max_stride=stride)
        X_test, y_test, _ = make_feature_vectors_version_seven(trackedObjects=tracks_test, labels=labels_test, max_stride=stride)

    models = {
        'KNN' : KNeighborsClassifier,
        #'GP' : GaussianProcessClassifier,
        'GNB' : GaussianNB,
        'MLP' : MLPClassifier,
        'SGD_modified_huber' : SGDClassifier,
        'SGD_log_loss' : SGDClassifier,
        'SVM' : SVC,
        'DT' : DecisionTreeClassifier
    }
    
    parameters = [{
                    'KNN' : {'n_neighbors' : 15},
                    'GP' :  {},
                    'GNB' : {},
                    'MLP' : {'max_iter' : 5000, 'solver' : 'sgd'},
                    'SGD_modified_huber' : {'loss' : 'modified_huber'},
                    'SGD_log_loss' : {'loss' : 'log_loss'},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True, 'max_iter' : 26000},
                    'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 3},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 2000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 2000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 2000},
                    'SVM' : {'kernel' : 'linear', 'probability' : True,'max_iter': 2000},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 1},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 3000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 3000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 3000},
                    'SVM' : {'kernel' : 'linear', 'probability' : True, 'max_iter': 4000},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 7},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 4000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 16000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 16000},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True, 'max_iter': 8000},
                    #'DT' : {} 
                }]
    
    splits = np.append(np.arange(1,6,1), ["Max split", "Mean", "Standart deviation"])
    basic_table = pd.DataFrame()
    balanced_table = pd.DataFrame()
    top_1_table = pd.DataFrame()
    top_2_table = pd.DataFrame()
    top_3_table = pd.DataFrame()
    final_test_basic = pd.DataFrame()
    final_test_balanced = pd.DataFrame()
    final_test_top_k_idx = ["Top_1", "Top_2", "Top_3"]
    final_test_top_k = pd.DataFrame()

    basic_table["Split"] = splits
    balanced_table["Split"] = splits
    top_1_table["Split"] = splits
    top_2_table["Split"] = splits
    top_3_table["Split"] = splits
    final_test_top_k["Top"] = final_test_top_k_idx

    parameters_table = pd.DataFrame(parameters[estimator_params_set-1])

    # makeing top_k scorer callables, to be able to set their k parameter
    #top_1_scorer = make_scorer(top_k_accuracy_score, k=1)
    #top_2_scorer = make_scorer(top_k_accuracy_score, k=2)
    #top_3_scorer = make_scorer(top_k_accuracy_score, k=3)
    top_k_scorers = {
        'top_1' : make_scorer(top_k_accuracy_score, k=1, needs_proba=True),
        'top_2' : make_scorer(top_k_accuracy_score, k=2, needs_proba=True),
        'top_3' : make_scorer(top_k_accuracy_score, k=3, needs_proba=True) 
    }

    print(f"\nTraining dataset size: {X_train.shape[0]}")
    print(f"Validation dataset size: {X_test.shape[0]}\n")

    print(f"Number of clusters: {len(set(labels_train))}")

    trained_classifiers = []

    t1 = time.time()
    for m in tqdm(models, desc="Cross validate models"):
        clf_ovr = OneVSRestClassifierExtended(estimator=models[m](**parameters[estimator_params_set-1][m]), tracks=tracks_train, n_jobs=n_jobs)

        basic_scores = cross_val_score(clf_ovr, X_train, y_train, cv=n_splits, fit_params=fit_params, n_jobs=n_jobs)
        basic_table[m] = np.append(basic_scores, [np.max(basic_scores), basic_scores.mean(), basic_scores.std()]) 

        balanced_scores = cross_val_score(clf_ovr, X_train, y_train, cv=n_splits, scoring='balanced_accuracy', fit_params=fit_params, n_jobs=n_jobs)
        balanced_table[m] = np.append(balanced_scores, [np.max(balanced_scores), balanced_scores.mean(), balanced_scores.std()]) 

        top_k_scores = cross_validate(clf_ovr, X_train, y_train, scoring=top_k_scorers, cv=5, fit_params=fit_params, n_jobs=n_jobs)
        top_1_table[m] = np.append(top_k_scores['test_top_1'], [np.max(top_k_scores['test_top_1']), top_k_scores['test_top_1'].mean(), top_k_scores['test_top_1'].std()])
        top_2_table[m] = np.append(top_k_scores['test_top_2'], [np.max(top_k_scores['test_top_2']), top_k_scores['test_top_2'].mean(), top_k_scores['test_top_2'].std()])
        top_3_table[m] = np.append(top_k_scores['test_top_3'], [np.max(top_k_scores['test_top_3']), top_k_scores['test_top_3'].mean(), top_k_scores['test_top_3'].std()])

        clf_ovr.fit(X_train, y_train, centroids=cluster_centroids)

        y_pred = clf_ovr.predict(X_test)
        y_pred_2 = clf_ovr.predict_proba(X_test)

        #final_balanced = clf.validate(X_test, y_test, threshold=0.5, centroids=cluster_centroids)
        #final_balanced_avg = np.average(final_balanced)
        #final_balanced_std = np.std(final_balanced)
        #final_test_balanced["Class"] = np.append(np.arange(len(final_balanced)), ["Mean", "Standart deviation"])
        #final_test_balanced[m] = np.append(final_balanced, [final_balanced_avg, final_balanced_std])

        final_top_k = []
        for i in range(1,4):
            final_top_k.append(top_k_accuracy_score(y_test, y_pred_2, k=i, labels=list(set(y_train))))
        final_test_top_k[m] = final_top_k

        final_basic = np.array([clf_ovr.score(X_test, y_test)])
        final_test_basic[m] = final_basic

        final_balanced = balanced_accuracy_score(y_test, y_pred)
        final_test_balanced[m] = np.array([final_balanced])

        trained_classifiers.append((m, clf_ovr))

    t2 = time.time()
    td = t2 - t1
    print("\n*Time: %d s*" % td)

    print("\n#### Classifier parameters\n")
    print(parameters[estimator_params_set-1])

    print("\n#### Cross-val Basic accuracy\n")
    print(basic_table.to_markdown())
    
    print("\n#### Cross-val Balanced accuracy\n")
    print(balanced_table.to_markdown())

    print("\n#### Cross-val Top 1 accuracy\n")
    print(top_1_table.to_markdown())

    print("\n#### Cross-val Top 2 accuracy\n")
    print(top_2_table.to_markdown())

    print("\n#### Cross-val Top 3 accuracy\n")
    print(top_3_table.to_markdown())

    print("\n#### Test set basic\n")
    print(final_test_basic.to_markdown())

    print("\n#### Test set balanced\n")
    print(final_test_balanced.to_markdown())

    print("\n#### Test set top k\n")
    print(final_test_top_k.to_markdown())

    print("\n#### Cross-val Basic accuracy\n")
    print(basic_table.to_latex())
    
    print("\n#### Cross-val Balanced accuracy\n")
    print(balanced_table.to_latex())

    print("\n#### Cross-val Top 1 accuracy\n")
    print(top_1_table.to_latex())

    print("\n#### Cross-val Top 2 accuracy\n")
    print(top_2_table.to_latex())

    print("\n#### Cross-val Top 3 accuracy\n")
    print(top_3_table.to_latex())

    print("\n#### Test set basic\n")
    print(final_test_basic.to_latex())

    print("\n#### Test set balanced\n")
    print(final_test_balanced.to_latex())

    print("\n#### Test set top k\n")
    print(final_test_top_k.to_latex())

    print("Writing tables to files...")
    if outputPath is not None:
        outputPath_Tables = Path(outputPath, "Tables")
        if not outputPath_Tables.exists():
            outputPath_Tables.mkdir(parents=True)
        with pd.ExcelWriter() as writer:
            parameters_table.to_excel(writer, sheet_name="Classifier parameters")
            basic_table.to_excel(writer, sheet_name="Cross Validation Basic scores")
            balanced_table.to_excel(writer, sheet_name="Cross Validation Balanced scores")
            top_1_table.to_excel(writer, sheet_name="Cross Validation Top 1 scores")
            top_2_table.to_excel(writer, sheet_name="Cross Validation Top 2 scores")
            top_3_table.to_excel(writer, sheet_name="Cross Validation Top 3 scores")
            final_test_basic.to_excel(writer, sheet_name="Validation set Basic scores")
            final_test_balanced.to_excel(writer, sheet_name="Validation set Balanced scores")
            final_test_top_k.to_excel(writer, sheet_name="Validation set Top K scores")
        
    if outputPath is not None:
        outputPath_classifiers = Path(outputPath, "Models")
        if not outputPath_classifiers.exists():
            outputPath_classifiers.mkdir(parents=True)
        for i, clf in enumerate(trained_classifiers):
            print("Saving model %s..." % clf[0])
            save_model(str(Path(outputPath_classifiers)), clf[0], clf[1])

    print()
    return basic_table, balanced_table, top_1_table, top_2_table, top_3_table, final_test_basic, final_test_balanced, final_test_top_k

def cross_validate_multiclass(path2dataset: str, outputPath: str = None, train_ratio=0.75, seed=1, n_splits=5, n_jobs=18, estimator_params_set=1, classification_features_version: str = "v1", stride: int = 15, level: float = None, n_weights: int = 3, weights_preset: int = 1, threshold: float = 0.7, **estkwargs):
    """Run cross validation on chosen classifiers with different feature vectors.

    Args:
        path2dataset (str): dataset path.
        outputPath (str, optional): The path to the output table to save the results. Defaults to None.
        train_ratio (float, optional): The ratio of the training dataset compared to the test dataset. Defaults to 0.75.
        seed (int, optional): Seed value to be able reproduce shuffle on dataset. Defaults to 1.
        n_splits (int, optional): Cross validation split number. Defaults to 5.
        n_jobs (int, optional): Number of paralell processes to run. Defaults to 18.
        estimator_params_set (int, optional): Choose classifier parameter set. Defaults to 1.
        classification_features_version (str, optional): Choose which feature vector to use for classification. Defaults to "v1".
        stride (int, optional): Size of the sliding window used to generate feature vectors from detection history. Defaults to 15.
        level (bool, optional): Choose if dataset should be balanced or stay as it is after enrichment. Defaults to False.
        n_weights (int, optional): The number of dimensions, that is going to be added between the first and last dimension in the feature vector. Defaults to n_weights.

    Returns:
        tuple: cross validation results in pandas datastructure 
    """
    from clustering import clustering_on_feature_vectors
    from sklearn.cluster import OPTICS
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from classifier import OneVSRestClassifierExtended
    from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
    from sklearn.metrics import top_k_accuracy_score, make_scorer, balanced_accuracy_score
    from visualizer import aoiextraction

    tracks = load_dataset(path2dataset)

    # cluster tracks
    tracks_filtered = filter_trajectories(trackedObjects=tracks, threshold=threshold)
    cls_samples = make_4D_feature_vectors(tracks_filtered)
    _, labels = clustering_on_feature_vectors(X=cls_samples, estimator=OPTICS, n_jobs=n_jobs, **estkwargs)
    tracks_labeled = tracks_filtered[labels > -1]
    cluster_labels = labels[labels > -1]

    train_dict = []
    for i in range(len(tracks_labeled)):
        train_dict.append({
            "track" : tracks_labeled[i],
            "class" : cluster_labels[i]
        })

    # shuffle tracks, and separate into a train and test dataset
    train, test = train_test_split(train_dict, train_size=train_ratio, random_state=seed)

    tracks_train = [t["track"] for t in train]
    labels_train = np.array([t["class"] for t in train])
    tracks_test = [t["track"] for t in test]
    labels_test = np.array([t["class"] for t in test])

    cluster_centroids = None
    fit_params = None

    if classification_features_version == "v1":
        X_train, y_train, metadata_train = make_feature_vectors_version_one(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_one(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v1_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_one_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_one_half(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v2":
        X_train, y_train, metadata_train = make_feature_vectors_version_two(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_two(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v2_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_two_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_two_half(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v3":
        X_train, y_train, metadata_train = make_feature_vectors_version_three(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_three(trackedObjects=tracks_test, k=6, labels=labels_test)
        cluster_centroids = aoiextraction([t["track"] for t in tracks_filtered], [t["class"] for t in tracks_filtered]) 
        fit_params = {
            'centroids' : cluster_centroids
        }
    elif classification_features_version == "v3_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_three_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_three_half(trackedObjects=tracks_test, k=6, labels=labels_test)
        cluster_centroids = aoiextraction([t["track"] for t in tracks_filtered], [t["class"] for t in tracks_filtered]) 
        fit_params = {
            'centroids' : cluster_centroids
        }
    elif classification_features_version == "v4":
        X_train, y_train, metadata_train = make_feature_vectors_version_four(trackedObjects=tracks_train, max_stride=stride, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_four(trackedObjects=tracks_test, max_stride=stride, labels=labels_test)
    elif classification_features_version == "v5":
        X_train, y_train, metadata_train = make_feature_vectors_version_five(trackedObjects=tracks_train, labels=labels_train, max_stride=stride, n_weights=n_weights)
        X_test, y_test, metadata_test = make_feature_vectors_version_five(trackedObjects=tracks_test, labels=labels_test, max_stride=stride, n_weights=n_weights)
    elif classification_features_version == "v6":
        weights_presets = {
            1 : np.array([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0]),
            2 : np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        }
        X_train, y_train, metadata_train = make_feature_vectors_version_six(trackedObjects=tracks_train, labels=labels_train, max_stride=stride, weights=weights_presets[weights_preset])
        X_test, y_test, metadata_test = make_feature_vectors_version_six(trackedObjects=tracks_test, labels=labels_test, max_stride=stride, weights=weights_presets[weights_preset])
    if level is not None:
        X_train, y_train = level_features(X_train, y_train, level)
        X_test, y_test = level_features(X_test, y_test, level)
    elif classification_features_version == "v7":
        X_train, y_train, _ = make_feature_vectors_version_seven(trackedObjects=tracks_train, labels=labels_train, max_stride=stride)
        X_test, y_test, _ = make_feature_vectors_version_seven(trackedObjects=tracks_test, labels=labels_test, max_stride=stride)

    models = {
        'KNN' : KNeighborsClassifier,
        #'GP' : GaussianProcessClassifier,
        #'GNB' : GaussianNB,
        #'MLP' : MLPClassifier,
        #'SGD_modified_huber' : SGDClassifier,
        #'SGD_log_loss' : SGDClassifier,
        'SVM' : SVC,
        'DT' : DecisionTreeClassifier
    }
    
    parameters = [{
                    'KNN' : {'n_neighbors' : 15},
                    'GP' :  {},
                    'GNB' : {},
                    'MLP' : {'max_iter' : 26000, 'solver' : 'sgd'},
                    'SGD_modified_huber' : {'loss' : 'modified_huber'},
                    'SGD_log_loss' : {'loss' : 'log_loss'},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True, 'max_iter' : 26000},
                    'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 3},
                    'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 2000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 2000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 2000},
                    'SVM' : {'kernel' : 'linear', 'probability' : True,'max_iter': 2000},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 1},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 3000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 3000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 3000},
                    'SVM' : {'kernel' : 'linear', 'probability' : True, 'max_iter': 4000},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 7},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 4000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 16000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 16000},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True, 'max_iter': 8000},
                    #'DT' : {} 
                }]
    
    splits = np.append(np.arange(1,6,1), ["Max split", "Mean", "Standart deviation"])
    basic_table = pd.DataFrame()
    balanced_table = pd.DataFrame()
    top_1_table = pd.DataFrame()
    top_2_table = pd.DataFrame()
    top_3_table = pd.DataFrame()
    final_test_basic = pd.DataFrame()
    final_test_balanced = pd.DataFrame()
    final_test_top_k_idx = ["Top_1", "Top_2", "Top_3"]
    final_test_top_k = pd.DataFrame()

    basic_table["Split"] = splits
    balanced_table["Split"] = splits
    top_1_table["Split"] = splits
    top_2_table["Split"] = splits
    top_3_table["Split"] = splits
    final_test_top_k["Top"] = final_test_top_k_idx

    parameters_table = pd.DataFrame(parameters[estimator_params_set-1])

    # makeing top_k scorer callables, to be able to set their k parameter
    #top_1_scorer = make_scorer(top_k_accuracy_score, k=1)
    #top_2_scorer = make_scorer(top_k_accuracy_score, k=2)
    #top_3_scorer = make_scorer(top_k_accuracy_score, k=3)
    top_k_scorers = {
        'top_1' : make_scorer(top_k_accuracy_score, k=1, needs_proba=True),
        'top_2' : make_scorer(top_k_accuracy_score, k=2, needs_proba=True),
        'top_3' : make_scorer(top_k_accuracy_score, k=3, needs_proba=True) 
    }

    print(f"\nTraining dataset size: {X_train.shape[0]}")
    print(f"Validation dataset size: {X_test.shape[0]}\n")

    print(f"Number of clusters: {len(set(labels_train))}")

    t1 = time.time()
    for m in tqdm(models, desc="Cross validate models"):
        clf = models[m](**parameters[estimator_params_set][m])

        basic_scores = cross_val_score(clf, X_train, y_train, cv=n_splits, n_jobs=n_jobs)
        basic_table[m] = np.append(basic_scores, [np.max(basic_scores), basic_scores.mean(), basic_scores.std()]) 

        balanced_scores = cross_val_score(clf, X_train, y_train, cv=n_splits, scoring='balanced_accuracy', n_jobs=n_jobs)
        balanced_table[m] = np.append(balanced_scores, [np.max(balanced_scores), balanced_scores.mean(), balanced_scores.std()]) 

        top_k_scores = cross_validate(clf, X_train, y_train, scoring=top_k_scorers, cv=5, n_jobs=n_jobs)
        top_1_table[m] = np.append(top_k_scores['test_top_1'], [np.max(top_k_scores['test_top_1']), top_k_scores['test_top_1'].mean(), top_k_scores['test_top_1'].std()])
        top_2_table[m] = np.append(top_k_scores['test_top_2'], [np.max(top_k_scores['test_top_2']), top_k_scores['test_top_2'].mean(), top_k_scores['test_top_2'].std()])
        top_3_table[m] = np.append(top_k_scores['test_top_3'], [np.max(top_k_scores['test_top_3']), top_k_scores['test_top_3'].mean(), top_k_scores['test_top_3'].std()])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_2 = clf.predict_proba(X_test)

        #final_balanced = clf.validate(X_test, y_test, threshold=0.5, centroids=cluster_centroids)
        #final_balanced_avg = np.average(final_balanced)
        #final_balanced_std = np.std(final_balanced)
        #final_test_balanced["Class"] = np.append(np.arange(len(final_balanced)), ["Mean", "Standart deviation"])
        #final_test_balanced[m] = np.append(final_balanced, [final_balanced_avg, final_balanced_std])

        final_top_k = []
        for i in range(1,4):
            final_top_k.append(top_k_accuracy_score(y_test, y_pred_2, k=i, labels=list(set(y_train))))
        final_test_top_k[m] = final_top_k

        final_basic = np.array([clf.score(X_test, y_test)])
        final_test_basic[m] = final_basic

        final_balanced = balanced_accuracy_score(y_test, y_pred)
        final_test_balanced[m] = np.array([final_balanced])

    t2 = time.time()
    td = t2 - t1
    print("\n*Time: %d s*" % td)

    print("\n#### Classifier parameters\n")
    print(parameters[estimator_params_set-1])

    print("\n#### Cross-val Basic accuracy\n")
    print(basic_table.to_markdown())
    
    print("\n#### Cross-val Balanced accuracy\n")
    print(balanced_table.to_markdown())

    print("\n#### Cross-val Top 1 accuracy\n")
    print(top_1_table.to_markdown())

    print("\n#### Cross-val Top 2 accuracy\n")
    print(top_2_table.to_markdown())

    print("\n#### Cross-val Top 3 accuracy\n")
    print(top_3_table.to_markdown())

    print("\n#### Test set basic\n")
    print(final_test_basic.to_markdown())

    print("\n#### Test set balanced\n")
    print(final_test_balanced.to_markdown())

    print("\n#### Test set top k\n")
    print(final_test_top_k.to_markdown())

    print("\n#### Cross-val Basic accuracy\n")
    print(basic_table.to_latex())
    
    print("\n#### Cross-val Balanced accuracy\n")
    print(balanced_table.to_latex())

    print("\n#### Cross-val Top 1 accuracy\n")
    print(top_1_table.to_latex())

    print("\n#### Cross-val Top 2 accuracy\n")
    print(top_2_table.to_latex())

    print("\n#### Cross-val Top 3 accuracy\n")
    print(top_3_table.to_latex())

    print("\n#### Test set basic\n")
    print(final_test_basic.to_latex())

    print("\n#### Test set balanced\n")
    print(final_test_balanced.to_latex())

    print("\n#### Test set top k\n")
    print(final_test_top_k.to_latex())

    if outputPath is not None:
        with pd.ExcelWriter(outputPath) as writer:
            parameters_table.to_excel(writer, sheet_name="Classifier parameters")
            basic_table.to_excel(writer, sheet_name="Cross Validation Basic scores")
            balanced_table.to_excel(writer, sheet_name="Cross Validation Balanced scores")
            top_1_table.to_excel(writer, sheet_name="Cross Validation Top 1 scores")
            top_2_table.to_excel(writer, sheet_name="Cross Validation Top 2 scores")
            top_3_table.to_excel(writer, sheet_name="Cross Validation Top 3 scores")
            final_test_basic.to_excel(writer, sheet_name="Validation set Basic scores")
            final_test_balanced.to_excel(writer, sheet_name="Validation set Balanced scores")
            final_test_top_k.to_excel(writer, sheet_name="Validation set Top K scores")

    print()
    return basic_table, balanced_table, top_1_table, top_2_table, top_3_table, final_test_basic, final_test_balanced, final_test_top_k

def calculate_metrics_exitpoints(dataset: str or List[str], 
                                 test_ratio: float, 
                                 output: str, threshold: float, 
                                 enter_exit_dist: float, 
                                 n_jobs: int, 
                                 models_to_benchmark: List[str], 
                                 mse_threshold: float = 0.5, 
                                 preprocessed: bool = False,
                                 test_trajectory_part: float = 1,
                                 background: str = None,
                                 **estkwargs):
    """Evaluate several one-vs-rest classifiers on the given dataset. 
    Recluster clusters based on exitpoint centroids and evaluate classifiers on the new clusters.

    Parameters:
        trainingSetPath (str | list[str]): Path to training dataset or list of paths to training datasets. 
        testingSetPath (str | list[str]): Path to testing dataset or list of paths to testing datasets. 
        output (str): Output directory path, where to save the results. Tables and Models. 
        threshold (float): Threshold for filtering trajectories. 
        n_jobs (int): Number of jobs to run in parallel. 
        mse_threshold (float, optional): MSE threshold for KMeans search. Defaults to 0.5.
    """
    from sklearn.cluster import OPTICS, Birch
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.metrics import top_k_accuracy_score, make_scorer, balanced_accuracy_score
    from clustering import (
        clustering_on_feature_vectors,
        kmeans_mse_clustering
    )
    from classifier import OneVSRestClassifierExtended

    # create output directories if output is given
    if output is not None:
        outputModels = Path(output)
        outputTables = Path(output, "Tables")
        if not outputModels.exists():
            outputModels.mkdir(parents=True)
        if not outputTables.exists():
            outputTables.mkdir(parents=True)

    # load datasets
    start = time.time()
    tracks = load_dataset(dataset)
    print("Dataset loaded in %d s" % (time.time() - start))
    print("Number of tracks: %d" % len(tracks))

    # cluster tracks
    start = time.time()
    if not preprocessed:
        tracks = filter_trajectories(tracks, threshold, enter_exit_dist, False, detDist=0.1)
    feature_vectors = make_4D_feature_vectors(tracks)
    print("Shape of feature vectors: {}".format(feature_vectors.shape))
    _, labels = clustering_on_feature_vectors(feature_vectors, 
                                              estimator=OPTICS, 
                                              n_jobs=n_jobs, 
                                              **estkwargs)
    print("Classes: {}".format(np.unique(labels)))
    # filter out not clustered tracks
    tracks_labeled = np.array(tracks)[labels > -1]
    cluster_labels = labels[labels > -1]
    print("Number of labeled trajectories after clustering: %d" % len(tracks_labeled))
    print("Clustering done in %d s" % (time.time() - start))
    # run clustering on cluster exit points centroids
    start = time.time()
    reduced_labels, _,_,_, centroids_labels = kmeans_mse_clustering(tracks_labeled, 
                                                           cluster_labels, 
                                                           n_jobs=n_jobs, 
                                                           mse_threshold=mse_threshold)
    print("Clustered exit centroids: {}".format(centroids_labels))
    print("Exit points clusters: {}".format(np.unique(reduced_labels)))
    print("Exit point clustering done in %d s" % (time.time() - start))

    cluster_centers = calc_cluster_centers(tracks_labeled, reduced_labels)

    # preprocess for train test split
    start = time.time()
    train_dict = []
    for i in range(len(tracks_labeled)):
        train_dict.append({
            "track" : tracks_labeled[i],
            "class" : cluster_labels[i],
            "reduced_class" : reduced_labels[i]
        })
    # shuffle tracks, and separate into a train and test dataset
    train, test = train_test_split(train_dict, test_size=test_ratio, random_state=42)
    # retrieve train and test tracks, labels and cluster centroid labels
    tracks_train = [t["track"] for t in train]
    labels_train = np.array([t["class"] for t in train])
    tracks_test = [t["track"] for t in test]
    labels_test = np.array([t["class"] for t in test])
    reduced_labels_train = np.array([t["reduced_class"] for t in train])
    reduced_labels_test = np.array([t["reduced_class"] for t in test])
    print("Train test split done in %d s" % (time.time() - start))
    print("Size of training set: %d" % len(tracks_train))
    print("Size of testing set: %d" % len(tracks_test))

    # Generate version one feature vectors for clustering
    start = time.time()
    X_train, y_train, metadata_train, y_reduced_train = make_feature_vectors_version_one(trackedObjects=tracks_train, k=6, labels=labels_train, reduced_labels=reduced_labels_train)
    X_test, y_test, metadata_test, y_reduced_test = make_feature_vectors_version_one(trackedObjects=tracks_test, k=6, labels=labels_test, reduced_labels=reduced_labels_test, up_until=test_trajectory_part)
    print("Feature vectors generated in %d s" % (time.time() - start))

    models = {
        "SVM" : SVC,
        "KNN" : KNeighborsClassifier,
        "DT" : DecisionTreeClassifier
    }

    parameters = [{
                    'KNN' : {'n_neighbors' : 15},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True, 'max_iter' : 26000},
                    'DT' : {} 
                }]

    # create top-k score functions
    """
    top_k_scorers = {
        'top_1' : make_scorer(top_k_accuracy_score, k=1, needs_proba=True),
        'top_2' : make_scorer(top_k_accuracy_score, k=2, needs_proba=True),
        'top_3' : make_scorer(top_k_accuracy_score, k=3, needs_proba=True) 
    }
    """

    original_results = pd.DataFrame(columns=['Top-1', 'Top-2', 'Top-3', 'Balanced Accuracy'])
    reduced_results = pd.DataFrame(columns=['Top-1', 'Top-2', 'Top-3', 'Balanced Accuracy']) 

    # train classifiers
    for m in models_to_benchmark:
        start = time.time()
        clf_ovr = OneVSRestClassifierExtended(estimator=models[m](**parameters[0][m]), n_jobs=n_jobs, centroid_labels=centroids_labels, centroid_coordinates=cluster_centers)
        clf_ovr.fit(X_train, y_train)
        # predict probabilities
        y_pred = clf_ovr.predict(X_test)
        y_pred_proba = clf_ovr.predict_proba(X_test)
        # convert y_pred to cluster centroid labels
        y_pred_reduced = np.array([centroids_labels[y] for y in y_pred])
        y_pred_proba_reduced = clf_ovr.predict_proba(X_test, centroids_labels)
        print("Classifier %s trained in %d s" % (m, time.time() - start))
        # evaluate based on original clusters 
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        final_top_k = []
        for i in range(1,4):
            final_top_k.append(top_k_accuracy_score(y_test, y_pred_proba, k=i, labels=list(set(y_train))))
        # add results to table
        original_results.loc[m] = {'Top-1': final_top_k[0], 'Top-2': final_top_k[1], 'Top-3': final_top_k[2], 'Balanced Accuracy': balanced_accuracy}
        print("Classifier %s evaluation based on original clusters: balanced accuracy: %f, top-1: %f, top-2: %f, top-3: %f" % (m, balanced_accuracy, final_top_k[0], final_top_k[1], final_top_k[2]))
        # evaluate based on exit centroids 
        balanced_accuracy_centroids = balanced_accuracy_score(y_reduced_test, y_pred_reduced)
        final_top_k_centroids = []
        for i in range(1,4):
            final_top_k_centroids.append(top_k_accuracy_score(y_reduced_test, y_pred_proba_reduced, k=i, labels=list(set(y_train))))
        # add results to table
        reduced_results.loc[m] = {'Top-1': final_top_k_centroids[0], 'Top-2': final_top_k_centroids[1], 'Top-3': final_top_k_centroids[2], 'Balanced Accuracy': balanced_accuracy_centroids}
        print("Classifier %s evaluation based on exit point centroids: balanced accuracy: %f, top-1: %f, top-2: %f, top-3: %f" % (m, balanced_accuracy_centroids, final_top_k_centroids[0], final_top_k_centroids[1], final_top_k_centroids[2]))
        ### Save model if output path is given ###
        if output is not None:
            misclassified = list({metadata_test[i][-1] for i in range(len(X_test)) if y_test[i] != y_pred[i]})
            save_model(outputModels, m, clf_ovr)
            paths = {o._dataset for o in misclassified}
            for p in paths:
                _misclassified = []
                for j in range(len(misclassified)):
                    if misclassified[j]._dataset == p:
                        _misclassified.append(misclassified[j])
                _output = Path(output) / "misclassified"
                _output.mkdir(parents=True, exist_ok=True)
                print(save_trajectories(_misclassified, _output, f"{m}_{str(Path(p).stem)}"))

        
    ### Print out tables ###
    print()
    print(original_results.to_markdown())
    print()
    print(reduced_results.to_markdown())
    print()
    
# submodule functions
def train_binary_classifiers_submodule(args):
    train_binary_classifiers(args.database, args.outdir, 
                            min_samples=args.min_samples, 
                            max_eps=args.max_eps,xi=args.xi, 
                            min_cluster_size=args.min_samples, n_jobs=args.n_jobs,
                            cluster_features_version=args.cluster_features_version,
                            classification_features_version=args.classification_features_version,
                            stride=args.stride,
                            batch_size=args.batchsize,
                            level=args.level, n_weights=args.n_weights,
                            weights_preset=args.weights_preset,
                            threshold=args.threshold,
                            p_norm=args.p_norm)

def cross_validation_submodule(args):
    cross_validate(args.database, 
                args.output, 
                args.train_ratio, 
                args.seed, 
                n_jobs=args.n_jobs, 
                estimator_params_set=args.param_set, 
                #cluster_features_version=args.cluster_features_version,
                classification_features_version=args.classification_features_version,
                stride=args.stride, 
                level=args.level, 
                n_weights=args.n_weights, 
                weights_preset=args.weights_preset,
                threshold=args.threshold,
                min_samples=args.min_samples, # clustering param
                max_eps=args.max_eps, # clustering param
                xi=args.xi, # clustering param
                p=args.p_norm) # clustering param

def cross_validation_multiclass_submodule(args):
    cross_validate_multiclass(args.database, 
                args.output, 
                args.train_ratio, 
                args.seed, 
                n_jobs=args.n_jobs, 
                estimator_params_set=args.param_set, 
                #cluster_features_version=args.cluster_features_version,
                classification_features_version=args.classification_features_version,
                stride=args.stride, 
                level=args.level, 
                n_weights=args.n_weights, 
                weights_preset=args.weights_preset,
                threshold=args.threshold,
                min_samples=args.min_samples, # clustering param
                max_eps=args.max_eps, # clustering param
                xi=args.xi, # clustering param
                p=args.p_norm) # clustering param

def investigate_renitent_features(args):
    investigateRenitent(args.model, args.threshold, 
                        min_samples=args.min_samples, 
                        max_eps=args.max_eps, xi=args.xi, 
                        min_cluster_size=args.min_samples, 
                        n_jobs=args.n_jobs)

def plot_module(args):
    plot_decision_tree(args.decision_tree)

def exitpoint_metric_module(args):
    calculate_metrics_exitpoints(
        dataset=args.dataset, 
        test_ratio=args.test, 
        output=args.output,
        threshold=args.threshold,
        preprocessed=args.preprocessed,
        enter_exit_dist=args.enter_exit_distance,
        n_jobs=args.n_jobs,
        models_to_benchmark=args.models,
        mse_threshold=args.mse,
        test_trajectory_part=args.test_part,
        background=args.background,
        min_samples=args.min_samples,
        max_eps=args.max_eps,
        xi=args.xi,
        p=args.p_norm
    )

def main():
    import argparse
    argparser = argparse.ArgumentParser("Train, validate and test for renitent detection.")
    argparser.add_argument("--n-jobs", type=int, help="Number of processes.", default=1)

    submodule_parser = argparser.add_subparsers(help="Program functionalities.")

    # add subcommands for training binary classifiers
    # train_binary_classifiers_parser = submodule_parser.add_parser(
    #     "train",
    #     help="Run Classification on dataset, but not as a multi class classification, rather do "
    #          "binary classification for each cluster."
    # )
    # train_binary_classifiers_parser.add_argument("-db", "--database", help="Path to database file. This should be an unclustered joblib dataset file.", type=str)
    # train_binary_classifiers_parser.add_argument("--outdir", "-o", help="Output directory path.", type=str)
    # train_binary_classifiers_parser.add_argument("--min_samples", default=10, type=int, 
    #     help="OPTICS parameter: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    # train_binary_classifiers_parser.add_argument("--max_eps", type=float, default=0.2, 
    #     help="OPTICS parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    # train_binary_classifiers_parser.add_argument("--xi", type=float, default=0.15, 
    #     help="OPTICS parameter: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.")
    # train_binary_classifiers_parser.add_argument("--min_cluster_size", default=10, type=float,
    #     help="OPTICS parameter: Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).")
    # train_binary_classifiers_parser.add_argument("--cluster_features_version", choices=["4D", "6D"], help="Choose which version of features to use for clustering.", default="6D")
    # train_binary_classifiers_parser.add_argument("--classification_features_version", choices=["v1", "v1_half", "v2", "v2_half", "v3", "v3_half", "v4", "v5", "v6", "v7", "v8", "v9"], help="Choose which version of features to use for classification.", default="v1")
    # train_binary_classifiers_parser.add_argument("--stride", default=15, type=int, help="Set stride value of classification features v4.")
    # train_binary_classifiers_parser.add_argument("--batchsize", type=int, default=None, help="Set training batch size.")
    # train_binary_classifiers_parser.add_argument("--level", default=None, type=float, help="Use this flag to set the level ratio of the samples count balancer function.")
    # train_binary_classifiers_parser.add_argument("--n_weights", default=3, type=int, help="The number of dimensions to add into the feature vector, between the first and the last dimension.")
    # train_binary_classifiers_parser.add_argument("--weights_preset", choices=[1, 2], type=int, default=1, help="Choose the weight vector. 1 = [1.,1.,1.,1.,1.5,1.5,1.5,1.5,2.,2.,2.,2.], 2 = [1.,1.,1.,1.,2.,2.,2.,2.,3.,3.,3.,3.]")
    # train_binary_classifiers_parser.add_argument("--threshold", type=float, default=0.4, help="Threshold value for clustering.")
    # train_binary_classifiers_parser.add_argument("-p", "--p_norm", default=2, type=int, help="P parameter of the clustering algorithm.")
    # train_binary_classifiers_parser.set_defaults(func=train_binary_classifiers_submodule)

    # # add subcommands for cross validating classifiers 
    # cross_validation_parser = submodule_parser.add_parser(
    #     "cross-validation",
    #     help="Run cross validation with given dataset."
    # )
    # cross_validation_parser.add_argument("-db", "--database", help="Path to database file. This should be an already clustered joblib dataset file.", type=str)
    # cross_validation_parser.add_argument("--output", "-o", help="Output file path, make sure that the directory of the outputted file exists.", type=str)
    # cross_validation_parser.add_argument("--train_ratio", help="Size of the train dataset. (0-1 float)", type=float, default=0.75)
    # cross_validation_parser.add_argument("--seed", help="Seed for random number generator to be able to reproduce dataset shuffle.", type=int, default=1)
    # cross_validation_parser.add_argument("--param_set", help="Choose between the parameter sets that will be given to the classifiers.", type=int, choices=[1,2,3,4], default=1)
    # cross_validation_parser.add_argument("--classification_features_version", choices=["v1", "v1_half", "v2", "v2_half", "v3", "v3_half", "v4", "v5", "v6", "v7", "v8", "v9"], help="Choose which version of features to use for classification.", default="v1")
    # cross_validation_parser.add_argument("--stride", default=15, type=int, help="Set stride value of classification features v4.")
    # #cross_validation_parser.add_argument("--cluster_features_version", choices=["4D", "6D"], help="Choose which version of features to use for clustering.", default="6D")
    # cross_validation_parser.add_argument("--level", default=None, type=float, help="Use this flag to set the level ratio of the samples count balancer function.")
    # cross_validation_parser.add_argument("--n_weights", default=3, type=int, help="The number of dimensions to add into the feature vector, between the first and the last dimension.")
    # cross_validation_parser.add_argument("--weights_preset", choices=[1, 2], type=int, default=1, help="Choose the weight vector. 1 = [1.,1.,1.,1.,1.5,1.5,1.5,1.5,2.,2.,2.,2.], 2 = [1.,1.,1.,1.,2.,2.,2.,2.,3.,3.,3.,3.]")
    # cross_validation_parser.add_argument("--threshold", default=0.7, type=float, help="Threshold value for min max filtering of tracked object.")
    # cross_validation_parser.add_argument("--min_samples", default=50, type=int, help="OPTICS clustering param.")
    # cross_validation_parser.add_argument("--max_eps", default=0.2, type=float, help="OPTICS clustering param.")
    # cross_validation_parser.add_argument("--xi", default=0.15, type=float, help="OPTICS clustering param.")
    # cross_validation_parser.add_argument("-p", "--p_norm", default=0.15, type=float, help="OPTICS clustering param.")
    # cross_validation_parser.set_defaults(func=cross_validation_submodule)

    # # add subcommands for renitent investigation module
    # renitent_filter_parser = submodule_parser.add_parser(
    #     "renitent-filter",
    #     help="Look at detections, that cant be predicted above a given threshold value."
    # )
    # renitent_filter_parser.add_argument("--model", help="Trained classifier.", type=str)
    # renitent_filter_parser.add_argument("--threshold", type=float, default=0.5, help="Balanced accuracy threshold.")
    # renitent_filter_parser.add_argument("--min_samples", default=10, type=int, 
    #     help="OPTICS parameter: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    # renitent_filter_parser.add_argument("--max_eps", type=float, default=0.2,
    #     help="OPTICS parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    # renitent_filter_parser.add_argument("--xi", type=float, default=0.15,
    #     help="OPTICS parameter: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.")
    # renitent_filter_parser.add_argument("--min_cluster_size", default=10, type=float, 
    #     help="OPTICS parameter: Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction" 
    #          "of the number of samples (rounded to be at least 2).")
    # renitent_filter_parser.set_defaults(func=investigate_renitent_features)

    # cross_validation_multiclass_parser = submodule_parser.add_parser(
    #     "cross-val-multiclass",
    #     help="Run classification with stock multiclass models."
    # )
    # cross_validation_multiclass_parser.add_argument("-db", "--database", help="Path to database file. This should be an already clustered joblib dataset file.", type=str)
    # cross_validation_multiclass_parser.add_argument("--output", "-o", help="Output file path, make sure that the directory of the outputted file exists.", type=str)
    # cross_validation_multiclass_parser.add_argument("--train_ratio", help="Size of the train dataset. (0-1 float)", type=float, default=0.75)
    # cross_validation_multiclass_parser.add_argument("--seed", help="Seed for random number generator to be able to reproduce dataset shuffle.", type=int, default=1)
    # cross_validation_multiclass_parser.add_argument("--param_set", help="Choose between the parameter sets that will be given to the classifiers.", type=int, choices=[1,2,3,4], default=1)
    # cross_validation_multiclass_parser.add_argument("--classification_features_version", choices=["v1", "v1_half", "v2", "v2_half", "v3", "v3_half", "v4", "v5", "v6", "v7", "v8", "v9"], help="Choose which version of features to use for classification.", default="v1")
    # cross_validation_multiclass_parser.add_argument("--stride", default=15, type=int, help="Set stride value of classification features v4.")
    # cross_validation_multiclass_parser.add_argument("--level", default=None, type=float, help="Use this flag to set the level ratio of the samples count balancer function.")
    # cross_validation_multiclass_parser.add_argument("--n_weights", default=3, type=int, help="The number of dimensions to add into the feature vector, between the first and the last dimension.")
    # cross_validation_multiclass_parser.add_argument("--weights_preset", choices=[1, 2], type=int, default=1, help="Choose the weight vector. 1 = [1.,1.,1.,1.,1.5,1.5,1.5,1.5,2.,2.,2.,2.], 2 = [1.,1.,1.,1.,2.,2.,2.,2.,3.,3.,3.,3.]")
    # cross_validation_multiclass_parser.add_argument("--threshold", default=0.7, type=float, help="Threshold value for min max filtering of tracked object.")
    # cross_validation_multiclass_parser.add_argument("--min_samples", default=50, type=int, help="OPTICS clustering param.")
    # cross_validation_multiclass_parser.add_argument("--max_eps", default=0.2, type=float, help="OPTICS clustering param.")
    # cross_validation_multiclass_parser.add_argument("--xi", default=0.15, type=float, help="OPTICS clustering param.")
    # cross_validation_multiclass_parser.add_argument("-p", "--p_norm", default=0.15, type=float, help="OPTICS clustering param.")
    # cross_validation_multiclass_parser.set_defaults(func=cross_validation_multiclass_submodule)

    plot_parser = submodule_parser.add_parser(
        "plot-dt",
        help="Run plotting functions."
    )
    plot_parser.add_argument("--decision_tree", help="Path to decision tree joblib modell.")
    plot_parser.set_defaults(func=plot_module)

    exitpoint_metrics_parser = submodule_parser.add_parser(
        "train",
        help="Calculate metrics on only exitpoints."
    )
    exitpoint_metrics_parser.add_argument("--dataset", nargs="+",
                                          help="Dataset database path.")
    exitpoint_metrics_parser.add_argument("--test", type=float, default=0.2,
                                          help="Testset size in float. Default: 0.2")
    exitpoint_metrics_parser.add_argument("--test-part", type=float, default=1,
                                          help="Which part of the test set's trajectories should be used, 1/3, 2/3 or 3/3. Default: 1")
    exitpoint_metrics_parser.add_argument("--output", "-o", type=str, 
                                         help="Output files directory path. If not given, models wont be saved.")
    exitpoint_metrics_parser.add_argument("--threshold", type=float, default=0.7,
                                          help="Threshold value for clustering. Default: 0.7")
    exitpoint_metrics_parser.add_argument("--preprocessed", action="store_true", help="If dataset is already preprocessed, use this flag for faster loading. Default: False")
    exitpoint_metrics_parser.add_argument("--enter-exit-distance", type=float, default=0.4, help="Euclidean distance threshold for enter and exit points. Default: 0.4")
    exitpoint_metrics_parser.add_argument("--min-samples", default=50, type=int, help="OPTICS clustering param. Default: 50")
    exitpoint_metrics_parser.add_argument("--max-eps", default=0.2, type=float, help="OPTICS clustering param. Default: 0.2")
    exitpoint_metrics_parser.add_argument("--xi", default=0.15, type=float, help="OPTICS clustering param. Default: 0.15")
    exitpoint_metrics_parser.add_argument("-p", "--p-norm", default=2, type=float, help="OPTICS clustering param. Default: 2")
    exitpoint_metrics_parser.add_argument("--mse", default=0.5, type=float, help="Mean squared error threshold for KMeans search. Default: 0.5")
    exitpoint_metrics_parser.add_argument("--models", nargs="+", default=["SVM", "KNN", "DT"], help="Models to use for classification. Default: SVM, KNN, DT")
    exitpoint_metrics_parser.add_argument("--background", help="Background image for plots.")
    exitpoint_metrics_parser.set_defaults(func=exitpoint_metric_module)

    args = argparser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
