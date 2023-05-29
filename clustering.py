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
from processing_utils import (
    detectionParser, 
    trackedObjectFactory, 
    filter_out_false_positive_detections_by_enter_exit_distance, 
    filter_out_edge_detections, 
    filter_tracks, 
    make_2D_feature_vectors,
    make_4D_feature_vectors, 
    make_6D_feature_vectors,
    makeFeatureVectors_Nx2, 
    preprocess_database_data_multiprocessed, 
    shuffle_data,  
    load_dataset
)
from dataManagementClasses import Detection, TrackedObject
import databaseLoader
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os 
import tqdm

# the function below is deprectated, do not use
def affinityPropagation_on_featureVector(featureVectors: np.ndarray):
    """Run affinity propagation clustering algorithm on list of feature vectors. 

    Args:
        featureVector (list): A numpy ndarray of numpy ndarrays. ex.: [[x,y,x,y], [x2,y2,x2,y2]] 
    """
    from sklearn.cluster import AffinityPropagation 
    af= AffinityPropagation(preference=-50, random_state=0).fit(featureVectors)
    cluster_center_indices_= af.cluster_centers_indices_
    labels_ = af.labels_ 
    return labels_, cluster_center_indices_

# the function below is deprecated, do not use
def affinityPropagation_on_enter_and_exit_points(path2db: str, threshold: float):
    """Run affinity propagation clustering on first and last detections of objects.
    This way, the enter and exit areas on a videa can be determined.

    Args:
        path2db (str): Path to database file 
        threshold (float): Threshold value for filtering algorithm.
    """
    from itertools import cycle
    rawObjectData = databaseLoader.loadObjects(path2db)
    trackedObjects = []
    for rawObj in rawObjectData:
        tmpDets = []
        rawDets = databaseLoader.loadDetectionsOfObject(path2db, rawObj[0])
        for det in rawDets:
            tmpDets.append(detectionParser(det))
        trackedObjects.append(trackedObjectFactory(tmpDets))
    filteredTrackedObjects = filter_out_false_positive_detections_by_enter_exit_distance(trackedObjects, threshold)
    featureVectors = make_4D_feature_vectors(filteredTrackedObjects)
    labels, cluster_center_indices_= affinityPropagation_on_featureVector(featureVectors)
    n_clusters_= len(cluster_center_indices_)
    print("Estimated number of clusters: %d" % n_clusters_)
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    fig, axes = plt.subplots(2,1)
    axes[0].set_title("Enter points")
    axes[1].set_title("Exit points")
    for k, col in zip(range(n_clusters_), colors):
        axes[0].scatter(np.array([featureVectors[idx, 0] for idx in range(len(labels)) if labels[idx]==k]), 
        np.array([1-featureVectors[idx, 1] for idx in range(len(labels)) if labels[idx]==k]), c=col)
        axes[1].scatter(np.array([featureVectors[idx, 2] for idx in range(len(labels)) if labels[idx]==k]), 
        np.array([1-featureVectors[idx, 3] for idx in range(len(labels)) if labels[idx]==k]), c=col)
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_affinity_propagation_featureVectors_n_clusters_{n_clusters_}_threshold_{threshold}.png"
    fig.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

def k_means_on_featureVectors(featureVectors: np.ndarray, n_clusters: int):
    """Run kmeans clustrering algorithm on extracted feature vectors.

    Args:
        featureVectors (np.ndarray): a numpy array of extracted features 
        n_clusters (int): number of initial clusters for kmeans 

    Returns:
        np.ndarray: vector of labels, same length as given featureVectors vector 
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(featureVectors)
    return kmeans.labels_

def spectral_on_featureVectors(featureVectors: np.ndarray, n_clusters: int):
    """Run spectral clustering algorithm on extracted feature vectors.

    Args:
        featureVectors (np.ndarray): a numpy array of extracted features 
        n_clusters (int): number of initial clusters for spectral 

    Returns:
        np.ndarray: vector of labels, same length as given featureVectors vector 
    """
    from sklearn.cluster import SpectralClustering 
    spectral = SpectralClustering(n_clusters=n_clusters).fit(featureVectors)
    return spectral.labels_

def dbscan_on_featureVectors(featureVectors: np.ndarray, eps: float, min_samples: int, n_jobs: int):
    """Run dbscan clustering algorithm on extracted feature vectors.

    Args:
        featureVectors (np.ndarray): A numpy array of extracted features to run the clustering on. 
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.1.
        min_samples(int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        n_jobs(int, optional): The number of parallel jobs to run.

    Returns:
        labels: Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    """
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(featureVectors)
    return dbscan.labels_

def optics_on_featureVectors(featureVectors: np.ndarray, min_samples: int, xi: float, min_cluster_size: float, n_jobs: int, max_eps: float, p: int):
    """Run optics clustering algorithm on extracted feature vectors.

    Args:
        featureVectors (np.ndarray): A numpy array of extracted features to run the clustering on.
        min_samples (int, optional): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 10.
        xi (float, optional): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary. Defaults to 0.05.
        min_cluster_size (float, optional): Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2). If None, the value of min_samples is used instead. Defaults to 0.05.
        max_eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        p (int): Parameter for distance metric. Default = 2.

    Returns:
        labels: Cluster labels for each point in the dataset given to fit(). Noisy samples and points which are not included in a leaf cluster of cluster_hierarchy_ are labeled as -1.
    """
    from sklearn.cluster import OPTICS
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs).fit(featureVectors)
    return optics.labels_

def kmeans_clustering_on_nx2(path2db: str, n_clusters: int, threshold: float):
    """Run kmeans clustering on filtered feature vectors.

    Args:
        path2db (str): Path to database file. 
        n_clusters (int): number of initial clusters for kmeans 
        threshold (float): the threshold for the filtering algorithm 
    """
    filteredEnterDets, filteredExitDets = filter_out_false_positive_detections_by_enter_exit_distance(path2db, threshold)
    filteredEnterFeatures  = makeFeatureVectors_Nx2(filteredEnterDets)
    filteredExitFeatures = makeFeatureVectors_Nx2(filteredExitDets)
    colors = "bgrcmyk"
    labels_enter = k_means_on_featureVectors(filteredEnterFeatures, n_clusters)
    labels_exit = k_means_on_featureVectors(filteredExitFeatures, n_clusters)
    fig, axes = plt.subplots(n_clusters,1, figsize=(10,10))
    axes[0].set_xlim(0,2)
    axes[0].set_ylim(0,2)
    axes[1].set_xlim(0,2)
    axes[1].set_ylim(0,2)
    axes[0].set_title("Clusters of enter points")
    axes[1].set_title("Clusters of exit points")
    for i in range(n_clusters):
        enter_x = np.array([filteredEnterFeatures[idx][0] for idx in range(len(filteredEnterFeatures)) if labels_enter[idx]==i])
        enter_y = np.array([1-filteredEnterFeatures[idx][1] for idx in range(len(filteredEnterFeatures)) if labels_enter[idx]==i])
        axes[0].scatter(enter_x, enter_y, c=colors[i])
        exit_x = np.array([filteredExitFeatures[idx][0] for idx in range(len(filteredExitFeatures)) if labels_exit[idx]==i])
        exit_y = np.array([1-filteredExitFeatures[idx][1] for idx in range(len(filteredExitFeatures)) if labels_exit[idx]==i])
        axes[1].scatter(exit_x, exit_y, c=colors[i])
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_kmeans_n_cluster_{n_clusters}.png"
    fig.savefig(fname=os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi='figure', format='png')

def kmeans_clustering_on_nx4(trackedObjects: list, n_clusters: int, threshold: float, outdir: str, show=True):
    """Run kmeans clutering on N x 4 (x,y,x,y) feature vectors.

    Args:
        trackedObjects (list): List of object tracks. 
        n_clusters (int): Number of clusters. 
        threshold (float): Threshold value for the false positive filter algorithm. 
        outdir (str): Output directory path, where to save plotted images.
    """
    featureVectors = make_4D_feature_vectors(trackedObjects)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    labels = k_means_on_featureVectors(featureVectors, n_clusters)
    # create directory path name, where the plots will be saved
    dirpath = os.path.join(outdir, f"kmeans_on_nx4_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}")
    # check if dir exists
    if not os.path.isdir(dirpath):
        # make dir if not
        os.mkdir(dirpath)
    if n_clusters > 1:
        for i in range(n_clusters):
            fig, axes = plt.subplots(1,1,figsize=(10,10))
            trajectory_x = []
            trajectory_y = []
            for idx in range(len(featureVectors)):
                if labels[idx]==i:
                    for k in range(1,len(trackedObjects[idx].history)):
                        trajectory_x.append(trackedObjects[idx].history[k].X)
                        trajectory_y.append(1-trackedObjects[idx].history[k].Y)
            axes.scatter(trajectory_x, trajectory_y, s=2)
            axes.set_xlim(0,2)
            axes.set_ylim(0,2)   
            axes.set_title(f"Axis of cluster number {i}")
            enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==i])
            enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
            exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==i])
            exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
            axes.legend()
            axes.grid(True)
            if show:
                plt.show()
            # create filename
            filename = f"n_cluster_{i}.png"
            # save plot with filename into dir
            fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    else:
        print("Warning: n_clusters cant be 1, use heatmap instead. python3 dataAnalyzer.py -db <path_to_database> -hm")

def simple_kmeans_plotter(path2db:str, outdir: str, threshold:float, n_clusters:int, n_jobs=None):
    """Just plots and saves one clustering.

    Args:
        path2db (str): Path to database. 
        outdir (str): Output directory path, where to save plotted images.
        threshold (float): threshold value to filtering algorithm 
        n_clusters (int): number of clusters 
    """
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    filteredTracks = filter_out_edge_detections(tracks, threshold)
    kmeans_clustering_on_nx4(filteredTracks, n_clusters, threshold, outdir)

def kmeans_worker(path2db: str, outdir: str, threshold=(0.1, 0.7), k=(2,16), n_jobs=None):
    """This function automates the task of running kmeans clustering on different cluster numbers.

    Args:
        path2db (str): path to database file 
        outdir (str): Output directory path, where to save plotted images.
        n_cluster_start (int): starting number cluster 
        n_cluster_end (int): ending number cluster 
        threshold (float): threshold for filtering algorithm 

    Returns:
        bool: returns false if some crazy person uses the program 
    """
    if k[0] < 1 or k[1] < k[0]:
        print("Error: this is not how we use this program properly")
        return False
    trackedObjects = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    for i in range(k[0], k[1]+1): # plus 1 because range goes from k[0] to k[0]-1
        thres = threshold[0]
        while thres <= threshold[1]:
            filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
            kmeans_clustering_on_nx4(filteredTrackedObjects, i, thres, outdir=outdir, show=False)
            thres += 0.1

def spectral_clustering_on_nx4(trackedObjects: list, n_clusters: int, threshold: float, outdir: str, show=True):
    """Run spectral clustering on N x 4 (x,y,x,y) feature vectors.

    Args:
        trackedObjects (list[TrackedObjects]): Track dataset.
        outdir (str): Output directory path, where to save plotted images.
        n_clusters (int): Number of clusters. 
        threshold (float): Threshold value for the false positive filter algorithm. 
        outdir (str): Output directory path, where to save plotted images.
    """
    from sklearn.cluster import SpectralClustering 
    featureVectors = make_4D_feature_vectors(trackedObjects)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    # create directory path name, where the plots will be saved
    dirpath = os.path.join(outdir, f"spectral_on_nx4_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}")
    # check if dir exists
    if not os.path.isdir(dirpath):
        # make dir if not
        os.mkdir(dirpath)
    spec = SpectralClustering(n_clusters=n_clusters, n_jobs=-1).fit(featureVectors)
    labels = spec.labels_ 
    if n_clusters > 1:
        for i in range(n_clusters):
            fig, axes = plt.subplots(1,1, figsize=(10,10))
            trajectory_x = []
            trajectory_y = []
            for idx in range(len(featureVectors)):
                if labels[idx]==i:
                    for j in range(1,len(trackedObjects[idx].history)):
                        trajectory_x.append(trackedObjects[idx].history[j].X)
                        trajectory_y.append(1-trackedObjects[idx].history[j].Y)
            axes.scatter(trajectory_x, trajectory_y, s=2)
            axes.set_xlim(0,2)
            axes.set_ylim(0,2)   
            axes.set_title(f"Axis of cluster number {i}")
            enter_x = np.array([trackedObjects[idx].history[0].X for idx in range(len(featureVectors)) if labels[idx]==i])
            enter_y = np.array([1-trackedObjects[idx].history[0].Y for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
            exit_x = np.array([trackedObjects[idx].history[-1].X for idx in range(len(featureVectors)) if labels[idx]==i])
            exit_y = np.array([1-trackedObjects[idx].history[-1].Y for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
            axes.legend()
            axes.grid(True)
            if show:
                plt.show()
            # create filename
            filename = f"n_cluster_{i}.png"
            # save plot with filename into dir
            fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    else:
        print("Warning: n_clusters cant be 1, use heatmap instead. python3 dataAnalyzer.py -db <path_to_database> -hm")

def simple_spectral_plotter(path2db: str, outdir: str, threshold:float, n_clusters:int, n_jobs=None):
    """Create on spectral clustering plot with given parameters.

    Args:
        path2db (str): Path to datbase 
        outdir (str): Output directory path, where to save plotted images.
        threshold (float): threshold value for filtering algorithm 
        n_clusters (int): number of cluster 
    """
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    filteredTracks = filter_out_edge_detections(tracks, threshold)
    spectral_clustering_on_nx4(filteredTracks, n_clusters, threshold, outdir)

def spectral_worker(path2db: str, outdir: str, threshold=(0.1, 0.7), k=(2,16), n_jobs=None):
    """This function automates the task of running spectral clustering on different cluster numbers.

    Args:
        path2db (str): path to database file 
        outdir (str): Output directory path, where to save plotted images.
        n_cluster_start (int): starting number cluster 
        n_cluster_end (int): ending number cluster 
        threshold (float): threshold for filtering algorithm 

    Returns:
        bool: returns false if some crazy person uses the program 
    """
    if k[0] < 1 or k[1] < k[0]:
        print("Error: this is not how we use this program properly")
        return False
    trackedObjects = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    for i in range(k[0], k[1]+1): # plus 1 because range goes from k[0] to k[0]-1
        thres = threshold[0]
        while thres <= threshold[1]:
            filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
            spectral_clustering_on_nx4(filteredTrackedObjects, i, thres, outdir, show=False)
            thres += 0.1

def dbscan_clustering_on_nx4(trackedObjects: list, eps: float, min_samples: int, n_jobs: int, threshold: float, outdir: str, show=True, shuffle=False):
    """Run dbscan clustering on N x 4 (x,y,x,y) feature vectors.

    Args:
        trackedObjects (list): List of tracks. 
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.1.
        min_samples(int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        n_jobs(int, optional): The number of parallel jobs to run.
        threshold (float): Threshold value for filtering algorithm. 
        outdir (str): Output directory path, where to save plotted images.
        show (bool, optional): Boolean flag value to show plot or not. Defaults to True.
    """
    featureVectors = make_4D_feature_vectors(trackedObjects)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    labels = dbscan_on_featureVectors(featureVectors, eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if shuffle: # if shuffle flag was used, create a shuffle dir to place results in shuffle dir, to be able to compare shuffled and non shuffled results
        if not os.path.isdir(os.path.join(outdir, "shuffled")):
            os.mkdir(os.path.join(outdir, "shuffled"))
        # create directory path name, where the plots will be saved
        dirpath = os.path.join(outdir, "shuffled", f"dbscan_on_nx4_eps_{eps}_min_samples_{min_samples}_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}")
        # check if dir exists
        if not os.path.isdir(dirpath):
            # make dir if not
            os.mkdir(dirpath)
    else:
        # create directory path name, where the plots will be saved
        dirpath = os.path.join(outdir, f"dbscan_on_nx4_eps_{eps}_min_samples_{min_samples}_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}")
        # check if dir exists
        if not os.path.isdir(dirpath):
            # make dir if not
            os.mkdir(dirpath)
    if n_clusters > 1:
        for i in range(n_clusters):
            fig, axes = plt.subplots(1,1,figsize=(10,10))
            trajectory_x = []
            trajectory_y = []
            for idx in range(len(featureVectors)):
                if labels[idx]==i:
                    for k in range(1,len(trackedObjects[idx].history)):
                        trajectory_x.append(trackedObjects[idx].history[k].X)
                        trajectory_y.append(1-trackedObjects[idx].history[k].Y)
            axes.scatter(trajectory_x, trajectory_y, s=2)
            axes.set_xlim(0,2)
            axes.set_ylim(0,2)   
            axes.set_title(f"Axis of cluster number {i}")
            enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==i])
            enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
            exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==i])
            exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
            axes.legend()
            axes.grid(True)
            if show:
                plt.show()
            # create filename
            filename = f"n_cluster_{i}.png"
            # save plot with filename into dir
            fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    else:
        print("Warning: n_clusters cant be 1, use heatmap instead. python3 dataAnalyzer.py -db <path_to_database> -hm")

def simple_dbscan_plotter(path2db: str, threshold:float, eps: float, min_samples: int, n_jobs: int):
    """Run dbscan on dataset.

    Args:
        path2db (str): Path to database file. 
        threshold (float): Threshold for filtering algorithm. 
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        n_jobs (int): The number of parallel jobs to run.
    """
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs)
    tracksFiltered = filter_out_edge_detections(tracks, threshold)
    tracksFiltered = filter_tracks(tracksFiltered)
    dbscan_clustering_on_nx4(tracksFiltered, eps, min_samples, n_jobs, min_samples)

def dbscan_worker(path2db: str, outdir: str, eps: float, min_samples: int, n_jobs: int, threshold=(0.1, 0.7), k=(2,16), shuffle=False):
    """Run dbscan clustering on diffenrent threshold and n_cluster levels.

    Args:
        path2db (str): Path to database file. 
        outdir (str): Output directory path, where to save plotted images.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        n_jobs (int): The number of parallel jobs to run.
        threshold (tuple, optional): Threshold for filtering algorithm. Defaults to (0.1, 0.7).
        k (tuple, optional): n_cluster number. Defaults to (2,16).

    Returns:
        bool: Returns False of bad k parameters were given.
    """
    if k[0] < 1 or k[1] < k[0]:
        print("Error: this is not how we use this program properly")
        return False
    trackedObjects = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    if shuffle: # shuffle data to get different results
        shuffle_data(trackedObjects)
    for i in range(k[0], k[1]+1): # plus 1 because range goes from k[0] to k[0]-1
        thres = threshold[0]
        while thres <= threshold[1]:
            filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
            if shuffle:
                dbscan_clustering_on_nx4(trackedObjects=filteredTrackedObjects, threshold=thres, outdir=outdir, n_jobs=n_jobs, eps=eps, min_samples=min_samples, show=False, shuffle=True)
            else:
                dbscan_clustering_on_nx4(trackedObjects=filteredTrackedObjects, threshold=thres, outdir=outdir, n_jobs=n_jobs, eps=eps, min_samples=min_samples, show=False, shuffle=False)
            thres += 0.1

def optics_clustering_on_nx4(trackedObjects: list, min_samples: int, xi: float, min_cluster_size: float, max_eps:float, threshold: float, outdir: str, n_jobs=16, show=True):
    """Run optics clustering on N x 4 (x,y,x,y) feature vectors.

    Args:
        trackedObjects (list): List of tracks. 
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Also, up and down steep regions can`t have more than min_samples consecutive non-steep points.
        xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
        min_cluster_size (float): Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).
        max_eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        threshold (float): Threshold value for filtering algorithm.
        outdir (str): Output directory path, where to save plotted images.
        show (bool, optional): Boolean flag to show plot. Defaults to True.
    """
    featureVectors = make_4D_feature_vectors(trackedObjects)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    labels = optics_on_featureVectors(featureVectors, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters {n_clusters}")
    # create directory path name, where the plots will be saved
    dirpath = os.path.join(outdir, f"optics_on_nx4_min_samples_{min_samples}_max_eps_{max_eps}_xi_{xi}_min_cluster_size_{min_cluster_size}_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}")
    # check if dir exists
    if not os.path.isdir(dirpath):
        # make dir if not
        os.mkdir(dirpath)
    if n_clusters > 1:
        hist_data_overall = np.array([])
        fig_overall, axes_overall = plt.subplots(1,1,figsize=(20,10))
        for i in range(n_clusters):
            fig, axes = plt.subplots(1,2,figsize=(30,15))
            trajectory_x = []
            trajectory_y = []
            hist_data = []
            n_tracks = 0
            for idx in range(len(featureVectors)):
                if labels[idx]==i:
                    hist_length = len(trackedObjects[idx].history)
                    hist_data.append(hist_length)
                    n_tracks += 1
                    for k in range(1,len(trackedObjects[idx].history)):
                        trajectory_x.append(trackedObjects[idx].history[k].X)
                        trajectory_y.append(1-trackedObjects[idx].history[k].Y)
            axes[0].scatter(trajectory_x, trajectory_y, s=2)
            axes[0].set_xlim(0,2)
            axes[0].set_ylim(0,2)   
            axes[0].set_title(f"Axis of cluster number {i}, with {n_tracks} detections")
            enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==i])
            enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes[0].scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
            exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==i])
            exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes[0].scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
            axes[0].legend()
            axes[0].grid(True)
            hist_data = np.array(hist_data)
            hist_data_overall = np.append(hist_data_overall, hist_data)
            axes[1].hist(hist_data, bins="auto")
            axes[1].set_xlabel("Length of history")
            axes[1].set_ylabel("Number of objects")
            if show:
                plt.show()
            # create filename
            filename = f"n_cluster_{i}.png"
            # save plot with filename into dir
            fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
        hist_filename = "histogram.png"
        axes_overall.hist(hist_data_overall, bins="auto")
        axes_overall.set_xlabel("Length of history")
        axes_overall.set_ylabel("Number of objects")
        fig_overall.savefig(fname=os.path.join(dirpath, hist_filename), dpi='figure', format='png')
    else:
        print("Warning: n_clusters cant be 1, use heatmap instead. python3 dataAnalyzer.py -db <path_to_database> -hm")
    return labels

def simple_optics_plotter(path2db: str, outdir: str, min_samples=10, xi=0.05, threshold=0.3, min_cluster_size=0.05, max_eps=0.2, n_jobs=16):
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs)
    tracksFiltered = filter_out_edge_detections(tracks, threshold)
    tracksFiltered = filter_tracks(tracksFiltered)
    optics_clustering_on_nx4(tracksFiltered, min_samples, xi, min_cluster_size, threshold, max_eps, outdir)

def optics_worker(path2db: str, outdir: str, min_samples: int, xi: float, min_cluster_size: float, max_eps: float, threshold=(0.1, 0.7), k=(2,16), n_jobs=16):
    """Run dbscan clustering on diffenrent threshold and n_cluster levels.

    Args:
        path2db (str): Path to database file. 
        outdir (str): Output directory path, where to save plotted images.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
        min_cluster_size (float): Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).
        n_jobs (int): The number of parallel jobs to run.
        threshold (tuple, optional): Threshold for filtering algorithm. Defaults to (0.1, 0.7).
        k (tuple, optional): n_cluster number. Defaults to (2,16).

    Returns:
        bool: Returns False of bad k parameters were given.
    """
    if k[0] < 1 or k[1] < k[0]:
        print("Error: this is not how we use this program properly")
        return False
    trackedObjects = load_dataset(path2db)
    #trackedObjects = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    progress = 1
    thres_interval = 0.1
    #max_progress = k[1] * int(threshold[1] / thres_interval)
    max_progress = int(threshold[1] / thres_interval)
    #for i in range(k[0], k[1]+1): # plus 1 because range goes from k[0] to k[0]-1
    thres = threshold[0]
    while thres <= threshold[1]:
        filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
        optics_clustering_on_nx4(trackedObjects=filteredTrackedObjects, threshold=thres, outdir=outdir, n_jobs=n_jobs, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, max_eps=max_eps, show=False)
        thres += thres_interval 
        print(200 * '\n', '[', (progress-2) * '=', '>', int(max_progress-progress) * ' ', ']', flush=True)
        progress += 1

def cluster_optics_dbscan_on_featurevectors(featureVectors:np.ndarray, min_samples: int, xi: float, min_cluster_size: float, eps:float, n_jobs=-1):
    """Run clustering with optics, then dbscan using the results of the optics clustering.

    Args:
        featureVectors (np.ndarray): Numpy array of feature vectors, value of a feature vector can vary. 
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
        min_cluster_size (float): Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        n_jobs (int, optional): Number of processes to run simultaniously. Defaults to -1.

    Returns:
        list: list of labels
    """
    from sklearn.cluster import OPTICS, cluster_optics_dbscan
    clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs).fit(featureVectors)
    labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                    core_distances=clust.core_distances_, 
                                    ordering=clust.ordering_, eps=eps)
    return labels

def cluster_optics_dbscan_on_nx4(trackedObjects: list, min_samples: int, xi: float, min_cluster_size: float, eps:float, threshold: float, outdir: str, n_jobs=16, show=True):
    """Run optics clustering on N x 4 (x,y,x,y) feature vectors.

    Args:
        trackedObjects (list): List of tracks. 
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Also, up and down steep regions can`t have more than min_samples consecutive non-steep points.
        xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
        min_cluster_size (float): Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        threshold (float): Threshold value for filtering algorithm.
        path2db (str): Path to database file. 
        show (bool, optional): Boolean flag to show plot. Defaults to True.
    """
    featureVectors = make_4D_feature_vectors(trackedObjects)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    labels = cluster_optics_dbscan_on_featurevectors(featureVectors, min_samples, xi, min_cluster_size, eps, n_jobs)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # create directory path name, where the plots will be saved
    dirpath = os.path.join(outdir, f"opticsdbscan_on_nx4_min_samples_{min_samples}_eps_{eps}_xi_{xi}_min_cluster_size_{min_cluster_size}_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}")
    # check if dir exists
    if not os.path.isdir(dirpath):
        # make dir if not
        os.mkdir(dirpath)
    if n_clusters > 1:
        for i in range(n_clusters):
            fig, axes = plt.subplots(1,1,figsize=(10,10))
            trajectory_x = []
            trajectory_y = []
            n_tracks = 0
            for idx in range(len(featureVectors)):
                if labels[idx]==i:
                    n_tracks += 1
                    for k in range(1,len(trackedObjects[idx].history)):
                        trajectory_x.append(trackedObjects[idx].history[k].X)
                        trajectory_y.append(1-trackedObjects[idx].history[k].Y)
            axes.scatter(trajectory_x, trajectory_y, s=2)
            axes.set_xlim(0,2)
            axes.set_ylim(0,2)   
            axes.set_title(f"Axis of cluster number {i}, with {n_tracks} detections.")
            enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==i])
            enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
            exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==i])
            exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes.scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
            axes.legend()
            axes.grid(True)
            if show:
                plt.show()
            # create filename
            filename = f"n_cluster_{i}.png"
            # save plot with filename into dir
            fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    else:
        print("Warning: n_clusters cant be 1, use heatmap instead. python3 dataAnalyzer.py -db <path_to_database> -hm")

def cluster_optics_dbscan_plotter(path2db: str, outdir: str, min_samples=10, xi=0.05, threshold=0.3, min_cluster_size=0.05, eps=0.2, n_jobs=16):
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs)
    tracksFiltered = filter_out_edge_detections(tracks, threshold)
    tracksFiltered = filter_tracks(tracksFiltered)
    optics_clustering_on_nx4(tracksFiltered, min_samples, xi, min_cluster_size, threshold, eps, outdir)

def optics_dbscan_worker(path2db: str, outdir: str, min_samples=10, xi=0.05, min_cluster_size=0.05, eps=0.2, threshold=(0.1, 0.7), k=(2,16), n_jobs=16):
    """Run dbscan clustering on diffenrent threshold and n_cluster levels.

    Args:
        path2db (str): Path to database file. 
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        xi (float): Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.
        min_cluster_size (float): Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).
        n_jobs (int): The number of parallel jobs to run.
        threshold (tuple, optional): Threshold for filtering algorithm. Defaults to (0.1, 0.7).
        k (tuple, optional): n_cluster number. Defaults to (2,16).

    Returns:
        bool: Returns False of bad k parameters were given.
    """
    if k[0] < 1 or k[1] < k[0]:
        print("Error: this is not how we use this program properly")
        return False
    trackedObjects = load_dataset(path2db)
    #trackedObjects = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    progress = 1
    thres_interval = 0.1
    #max_progress = k[1] * int(threshold[1] / thres_interval)
    max_progress = int(threshold[1] / thres_interval)
    #for i in range(k[0], k[1]+1): # plus 1 because range goes from k[0] to k[0]-1
    thres = threshold[0]
    while thres <= threshold[1]:
        filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
        cluster_optics_dbscan_on_nx4(trackedObjects=filteredTrackedObjects, threshold=thres, outdir=outdir, n_jobs=n_jobs, min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size, eps=eps, show=False)
        thres += thres_interval 
        print(200 * '\n', '[', (progress-2) * '=', '>', int(max_progress-progress) * ' ', ']', flush=True)
        progress += 1

def clustering_on_feature_vectors(X: np.ndarray, estimator, n_jobs: int = -1, **estkwargs):
    """Run clustering with given estimator with given prameters.

    Args:
        X (np.ndarray): Feature vectors 
        estimator (estimator): sklearn estimator class
        n_jobs (int, optional): Number of processes to run. 
            Defaults to -1, that means all cpu threads will be utilized.

    Returns:
        ndarray: The labels ordered to the X features. 
    """
    try:
        cls = estimator(n_jobs=n_jobs, **estkwargs).fit(X)
    except:
        cls = estimator(**estkwargs).fit(X)
    return cls, cls.labels_

def clustering_on_2D_feature_vectors(estimator, trackedObjects: list[TrackedObject], outdir: str, n_jobs: int = -1, filter_threshold: float = None, **estkwargs):
    """Run clustering on 4 dimensional feature vectors.
    Create plots of every cluster with their tracks,
    and create histograms from the lenght of the tracks.

    Args:
        estimator (estimator): sklearn estimator class
        trackedObjects (list[TrackedObject]): TrackedObject python object dataset. 
        outdir (str): Output directory path. 
        n_jobs (int, optional): Number of processes to run. 
            Defaults to -1, that means all cpu threads will be utilized.
    """
    # Create feature vectors from list of trackedObjects
    X = make_2D_feature_vectors(trackedObjects)
    # Run the clustering estimator on X features with estkwargs
    fitted_estimator, Y = clustering_on_feature_vectors(X, estimator, n_jobs=n_jobs, **estkwargs)
    not_clustered_objects = np.array(trackedObjects) [ Y == -1 ] 
    X_nocluster = X [ Y == -1 ]
    Y_nocluster = Y [ Y == -1 ]
    clustered_objects = np.array(trackedObjects) [ Y > -1 ]
    X = X [ Y > -1 ]
    Y = Y [ Y > -1 ]
    labels = np.array(list(set(Y)))
    n_clusters = labels.shape[0]
    print(f"Number of trackedObjects: {len(trackedObjects)}")
    print(f"Number of feature vectors: {X.shape[0]}")
    print(f"Number of clusters identified: {n_clusters}")
    print(f"Labels: {labels}")

    # Generate dirname for plots
    dirpath = os.path.join(outdir, f"{fitted_estimator}_n_clusters-{n_clusters}_2D-feature-vectors-{X.shape[0]}_threshold-{filter_threshold}")
    # Create dir if it does not exists
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    histogram_X = np.array([])
    historgram_fig, histogram_axis = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

    for i in range(n_clusters):
        # Create figure for cluster trajectory and cluster histogram plot
        n_cluster_fig, n_cluster_axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
        # Declare the X for the i-th clusters historgram 
        n_cluster_histogram_X = [] 
        # Declare track counter for cluster
        n_tracks = 0
        # Declare arrays for x, y coordinates
        tracks_X = []
        tracks_Y = []
        for j in range(X.shape[0]):
            if Y[j] == i:
                n_cluster_histogram_x = len(clustered_objects[j].history)
                n_cluster_histogram_X.append( n_cluster_histogram_x ) 
                n_tracks += 1
                for k in range(1, n_cluster_histogram_x):
                    tracks_X.append(clustered_objects[j].history[k].X)
                    tracks_Y.append(clustered_objects[j].history[k].Y)

        n_cluster_axes[0].scatter(np.array(tracks_X), 1-np.array(tracks_Y), s=2)
        n_cluster_axes[0].set_xlim(0,2)
        n_cluster_axes[0].set_ylim(0,2)   
        n_cluster_axes[0].set_title(f"Axis of cluster number {i}, with {n_tracks} detections")
        #enter_x = np.array([X[idx, 0] for idx in range(X.shape[0]) if Y[idx]==i])
        #enter_y = np.array([1-X[idx, 1] for idx in range(X.shape[0]) if Y[idx]==i])
        #n_cluster_axes[0].scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
        exit_x = np.array([X[idx, 0] for idx in range(X.shape[0]) if Y[idx]==i])
        exit_y = np.array([1-X[idx, 1] for idx in range(X.shape[0]) if Y[idx]==i])
        n_cluster_axes[0].scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
        n_cluster_axes[0].legend()
        n_cluster_axes[0].grid(True)
        histogram_X= np.append(histogram_X, np.array(n_cluster_histogram_X))
        n_cluster_axes[1].hist(n_cluster_histogram_X, bins="auto", edgecolor="white")
        n_cluster_axes[1].set_xlabel("Length of history")
        n_cluster_axes[1].set_ylabel("Number of objects")
        # create filename
        filename = f"n_cluster_{i}_n_tracks_{n_tracks}.png"
        # save plot with filename into dir
        n_cluster_fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
        plt.close()

    # OPTICS given label -1 to those features that cant be ordered to any cluster
    nocluster_fig, nocluster_axis = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    no_cluster_tracks_X = []
    no_cluster_tracks_Y = []
    no_cluster_histogram_X = []
    for j in range(len(not_clustered_objects)):
        no_cluster_histogram_x = len(not_clustered_objects[j].history)
        no_cluster_histogram_X.append( no_cluster_histogram_x ) 
        for k in range(1, no_cluster_histogram_x):
            no_cluster_tracks_X.append(not_clustered_objects[j].history[k].X)
            no_cluster_tracks_Y.append(not_clustered_objects[j].history[k].Y)
    nocluster_axis[0].scatter(np.array(no_cluster_tracks_X), 1-np.array(no_cluster_tracks_Y), s=2)
    nocluster_axis[0].set_xlim(0,2)
    nocluster_axis[0].set_ylim(0,2)   
    nocluster_axis[0].set_title(f"Axis of cluster number -1, with {X_nocluster.shape[0]} detections")
    #no_cluster_enter_x = np.array([X_nocluster[idx, 0] for idx in range(X_nocluster.shape[0])])
    #no_cluster_enter_y = np.array([1-X_nocluster[idx, 1] for idx in range(X_nocluster.shape[0])])
    #nocluster_axis[0].scatter(no_cluster_enter_x, no_cluster_enter_y, c='g', s=10, label=f"Enter points")
    no_cluster_exit_x = np.array([X_nocluster[idx, 0] for idx in range(X_nocluster.shape[0])])
    no_cluster_exit_y = np.array([1-X_nocluster[idx, 1] for idx in range(X_nocluster.shape[0])])
    nocluster_axis[0].scatter(no_cluster_exit_x, no_cluster_exit_y, c='r', s=10, label=f"Exit points")
    nocluster_axis[0].legend()
    nocluster_axis[0].grid(True)
    nocluster_axis[1].hist(no_cluster_histogram_X, bins="auto", edgecolor="white")
    nocluster_axis[1].set_xlabel("Length of history")
    nocluster_axis[1].set_ylabel("Number of objects")
    # create filename
    filename = f"not_clustered_tracks_{X_nocluster.shape[0]}.png"
    # save plot with filename into dir
    nocluster_fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    plt.close()

    # Plot overall historgram
    hist_filename = "histogram.png"
    histogram_axis.hist(histogram_X, bins="auto", edgecolor="white")
    histogram_axis.set_xlabel("Length of history")
    histogram_axis.set_ylabel("Number of objects")
    historgram_fig.savefig(fname=os.path.join(dirpath, hist_filename), dpi='figure', format='png')
    plt.close()

def clustering_on_4D_feature_vectors(estimator, trackedObjects: list[TrackedObject], outdir: str, n_jobs: int = -1, filter_threshold: float = None, **estkwargs):
    """Run clustering on 4 dimensional feature vectors.
    Create plots of every cluster with their tracks,
    and create histograms from the lenght of the tracks.

    Args:
        estimator (estimator): sklearn estimator class
        trackedObjects (list[TrackedObject]): TrackedObject python object dataset. 
        outdir (str): Output directory path. 
        n_jobs (int, optional): Number of processes to run. 
            Defaults to -1, that means all cpu threads will be utilized.
    """
    # Create feature vectors from list of trackedObjects
    X = make_4D_feature_vectors(trackedObjects)
    # Run the clustering estimator on X features with estkwargs
    fitted_estimator, Y = clustering_on_feature_vectors(X, estimator, n_jobs=n_jobs, **estkwargs)
    not_clustered_objects = np.array(trackedObjects) [ Y == -1 ] 
    X_nocluster = X [ Y == -1 ]
    Y_nocluster = Y [ Y == -1 ]
    clustered_objects = np.array(trackedObjects) [ Y > -1 ]
    X = X [ Y > -1 ]
    Y = Y [ Y > -1 ]
    labels = np.array(list(set(Y)))
    n_clusters = labels.shape[0]
    print(f"Number of trackedObjects: {len(trackedObjects)}")
    print(f"Number of feature vectors: {X.shape[0]}")
    print(f"Number of clusters identified: {n_clusters}")
    print(f"Labels: {labels}")

    # Generate dirname for plots
    dirpath = os.path.join(outdir, f"{fitted_estimator}_n_clusters-{n_clusters}_4D-feature-vectors-{X.shape[0]}_threshold-{filter_threshold}")
    # Create dir if it does not exists
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    histogram_X = np.array([])
    historgram_fig, histogram_axis = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

    for i in range(n_clusters):
        # Create figure for cluster trajectory and cluster histogram plot
        n_cluster_fig, n_cluster_axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
        # Declare the X for the i-th clusters historgram 
        n_cluster_histogram_X = [] 
        # Declare track counter for cluster
        n_tracks = 0
        # Declare arrays for x, y coordinates
        tracks_X = []
        tracks_Y = []
        for j in range(X.shape[0]):
            if Y[j] == i:
                n_cluster_histogram_x = len(clustered_objects[j].history)
                n_cluster_histogram_X.append( n_cluster_histogram_x ) 
                n_tracks += 1
                for k in range(1, n_cluster_histogram_x):
                    tracks_X.append(clustered_objects[j].history[k].X)
                    tracks_Y.append(clustered_objects[j].history[k].Y)

        n_cluster_axes[0].scatter(np.array(tracks_X), 1-np.array(tracks_Y), s=2)
        n_cluster_axes[0].set_xlim(0,2)
        n_cluster_axes[0].set_ylim(0,2)   
        n_cluster_axes[0].set_title(f"Axis of cluster number {i}, with {n_tracks} detections")
        enter_x = np.array([X[idx, 0] for idx in range(X.shape[0]) if Y[idx]==i])
        enter_y = np.array([1-X[idx, 1] for idx in range(X.shape[0]) if Y[idx]==i])
        n_cluster_axes[0].scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
        exit_x = np.array([X[idx, 2] for idx in range(X.shape[0]) if Y[idx]==i])
        exit_y = np.array([1-X[idx, 3] for idx in range(X.shape[0]) if Y[idx]==i])
        n_cluster_axes[0].scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
        n_cluster_axes[0].legend()
        n_cluster_axes[0].grid(True)
        histogram_X= np.append(histogram_X, np.array(n_cluster_histogram_X))
        n_cluster_axes[1].hist(n_cluster_histogram_X, bins="auto", edgecolor="white")
        n_cluster_axes[1].set_xlabel("Length of history")
        n_cluster_axes[1].set_ylabel("Number of objects")
        # create filename
        filename = f"n_cluster_{i}_n_tracks_{n_tracks}.png"
        # save plot with filename into dir
        n_cluster_fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
        plt.close()

    # OPTICS given label -1 to those features that cant be ordered to any cluster
    nocluster_fig, nocluster_axis = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    no_cluster_tracks_X = []
    no_cluster_tracks_Y = []
    no_cluster_histogram_X = []
    for j in range(len(not_clustered_objects)):
        no_cluster_histogram_x = len(not_clustered_objects[j].history)
        no_cluster_histogram_X.append( no_cluster_histogram_x ) 
        for k in range(1, no_cluster_histogram_x):
            no_cluster_tracks_X.append(not_clustered_objects[j].history[k].X)
            no_cluster_tracks_Y.append(not_clustered_objects[j].history[k].Y)
    nocluster_axis[0].scatter(np.array(no_cluster_tracks_X), 1-np.array(no_cluster_tracks_Y), s=2)
    nocluster_axis[0].set_xlim(0,2)
    nocluster_axis[0].set_ylim(0,2)   
    nocluster_axis[0].set_title(f"Axis of cluster number -1, with {X_nocluster.shape[0]} detections")
    no_cluster_enter_x = np.array([X_nocluster[idx, 0] for idx in range(X_nocluster.shape[0])])
    no_cluster_enter_y = np.array([1-X_nocluster[idx, 1] for idx in range(X_nocluster.shape[0])])
    nocluster_axis[0].scatter(no_cluster_enter_x, no_cluster_enter_y, c='g', s=10, label=f"Enter points")
    no_cluster_exit_x = np.array([X_nocluster[idx, 2] for idx in range(X_nocluster.shape[0])])
    no_cluster_exit_y = np.array([1-X_nocluster[idx, 3] for idx in range(X_nocluster.shape[0])])
    nocluster_axis[0].scatter(no_cluster_exit_x, no_cluster_exit_y, c='r', s=10, label=f"Exit points")
    nocluster_axis[0].legend()
    nocluster_axis[0].grid(True)
    nocluster_axis[1].hist(no_cluster_histogram_X, bins="auto", edgecolor="white")
    nocluster_axis[1].set_xlabel("Length of history")
    nocluster_axis[1].set_ylabel("Number of objects")
    # create filename
    filename = f"not_clustered_tracks_{X_nocluster.shape[0]}.png"
    # save plot with filename into dir
    nocluster_fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    plt.close()

    # Plot overall historgram
    hist_filename = "histogram.png"
    histogram_axis.hist(histogram_X, bins="auto", edgecolor="white")
    histogram_axis.set_xlabel("Length of history")
    histogram_axis.set_ylabel("Number of objects")
    historgram_fig.savefig(fname=os.path.join(dirpath, hist_filename), dpi='figure', format='png')
    plt.close()

def clustering_on_6D_feature_vectors(estimator, trackedObjects: list[TrackedObject], outdir: str, n_jobs: int = -1, filter_threshold: float = None, **estkwargs):
    """Run clustering on 6 dimensional feature vectors.
    Create plots of every cluster with their tracks,
    and create histograms from the lenght of the tracks.

    Args:
        estimator (estimator): sklearn estimator class
        trackedObjects (list[TrackedObject]): TrackedObject python object dataset. 
        outdir (str): Output directory path. 
        n_jobs (int, optional): Number of processes to run. 
            Defaults to -1, that means all cpu threads will be utilized.
    """
    # Create feature vectors from list of trackedObjects
    X = make_6D_feature_vectors(trackedObjects)
    # Run the clustering estimator on X features with estkwargs
    fitted_estimator, Y = clustering_on_feature_vectors(X, estimator, n_jobs=n_jobs, **estkwargs)
    not_clustered_objects = np.array(trackedObjects) [ Y == -1 ] 
    X_nocluster = X [ Y == -1 ]
    Y_nocluster = Y [ Y == -1 ]
    clustered_objects = np.array(trackedObjects) [ Y > -1 ]
    X = X [ Y > -1 ]
    Y = Y [ Y > -1 ]
    labels = np.array(list(set(Y)))
    n_clusters = labels.shape[0]
    print(f"Number of trackedObjects: {len(trackedObjects)}")
    print(f"Number of feature vectors: {X.shape[0]}")
    print(f"Number of clusters identified: {n_clusters}")
    print(f"Labels: {labels}")

    # Generate dirname for plots
    dirpath = os.path.join(outdir, f"{fitted_estimator}_n_clusters-{n_clusters}_6D-feature-vectors-{X.shape[0]}_threshold-{filter_threshold}")
    # Create dir if it does not exists
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    histogram_X = np.array([])
    historgram_fig, histogram_axis = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

    for i in range(n_clusters):
        # Create figure for cluster trajectory and cluster histogram plot
        n_cluster_fig, n_cluster_axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
        # Declare the X for the i-th clusters historgram 
        n_cluster_histogram_X = [] 
        # Declare track counter for cluster
        n_tracks = 0
        # Declare arrays for x, y coordinates
        tracks_X = []
        tracks_Y = []
        for j in range(X.shape[0]):
            if Y[j] == i:
                n_cluster_histogram_x = len(clustered_objects[j].history)
                n_cluster_histogram_X.append( n_cluster_histogram_x ) 
                n_tracks += 1
                for k in range(1, n_cluster_histogram_x):
                    tracks_X.append(clustered_objects[j].history[k].X)
                    tracks_Y.append(clustered_objects[j].history[k].Y)

        n_cluster_axes[0].scatter(np.array(tracks_X), 1-np.array(tracks_Y), s=2)
        n_cluster_axes[0].set_xlim(0,2)
        n_cluster_axes[0].set_ylim(0,2)   
        n_cluster_axes[0].set_title(f"Axis of cluster number {i}, with {n_tracks} detections")
        enter_x = np.array([X[idx, 0] for idx in range(X.shape[0]) if Y[idx]==i])
        enter_y = np.array([1-X[idx, 1] for idx in range(X.shape[0]) if Y[idx]==i])
        n_cluster_axes[0].scatter(enter_x, enter_y, c='g', s=10, label=f"Enter points")
        mid_x = np.array([X[idx, 2] for idx in range(X.shape[0]) if Y[idx]==i])
        mid_y = np.array([1-X[idx, 3] for idx in range(X.shape[0]) if Y[idx]==i])
        n_cluster_axes[0].scatter(mid_x, mid_y, c='y', s=10, label=f"Middle points")
        exit_x = np.array([X[idx, 4] for idx in range(X.shape[0]) if Y[idx]==i])
        exit_y = np.array([1-X[idx, 5] for idx in range(X.shape[0]) if Y[idx]==i])
        n_cluster_axes[0].scatter(exit_x, exit_y, c='r', s=10, label=f"Exit points")
        n_cluster_axes[0].legend()
        n_cluster_axes[0].grid(True)
        histogram_X= np.append(histogram_X, np.array(n_cluster_histogram_X))
        n_cluster_axes[1].hist(n_cluster_histogram_X, bins="auto", edgecolor="white")
        n_cluster_axes[1].set_xlabel("Length of history")
        n_cluster_axes[1].set_ylabel("Number of objects")
        # create filename
        filename = f"n_cluster_{i}_n_tracks_{n_tracks}.png"
        # save plot with filename into dir
        n_cluster_fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
        plt.close()

    # OPTICS given label -1 to those features that cant be ordered to any cluster
    nocluster_fig, nocluster_axis = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
    no_cluster_tracks_X = []
    no_cluster_tracks_Y = []
    no_cluster_histogram_X = []
    for j in range(len(not_clustered_objects)):
        no_cluster_histogram_x = len(not_clustered_objects[j].history)
        no_cluster_histogram_X.append( no_cluster_histogram_x ) 
        for k in range(1, no_cluster_histogram_x):
            no_cluster_tracks_X.append(not_clustered_objects[j].history[k].X)
            no_cluster_tracks_Y.append(not_clustered_objects[j].history[k].Y)
    nocluster_axis[0].scatter(np.array(no_cluster_tracks_X), 1-np.array(no_cluster_tracks_Y), s=2)
    nocluster_axis[0].set_xlim(0,2)
    nocluster_axis[0].set_ylim(0,2)   
    nocluster_axis[0].set_title(f"Axis of cluster number -1, with {X_nocluster.shape[0]} detections")
    no_cluster_enter_x = np.array([X_nocluster[idx, 0] for idx in range(X_nocluster.shape[0])])
    no_cluster_enter_y = np.array([1-X_nocluster[idx, 1] for idx in range(X_nocluster.shape[0])])
    nocluster_axis[0].scatter(no_cluster_enter_x, no_cluster_enter_y, c='g', s=10, label=f"Enter points")
    no_cluster_mid_x = np.array([X_nocluster[idx, 2] for idx in range(X_nocluster.shape[0])])
    no_cluster_mid_y = np.array([1-X_nocluster[idx, 3] for idx in range(X_nocluster.shape[0])])
    nocluster_axis[0].scatter(no_cluster_mid_x, no_cluster_mid_y, c='y', s=10, label=f"Middle points")
    no_cluster_exit_x = np.array([X_nocluster[idx, 4] for idx in range(X_nocluster.shape[0])])
    no_cluster_exit_y = np.array([1-X_nocluster[idx, 5] for idx in range(X_nocluster.shape[0])])
    nocluster_axis[0].scatter(no_cluster_exit_x, no_cluster_exit_y, c='r', s=10, label=f"Exit points")
    nocluster_axis[0].legend()
    nocluster_axis[0].grid(True)
    nocluster_axis[1].hist(no_cluster_histogram_X, bins="auto", edgecolor="white")
    nocluster_axis[1].set_xlabel("Length of history")
    nocluster_axis[1].set_ylabel("Number of objects")
    # create filename
    filename = f"not_clustered_tracks_{X_nocluster.shape[0]}.png"
    # save plot with filename into dir
    nocluster_fig.savefig(fname=os.path.join(dirpath, filename), dpi='figure', format='png')
    plt.close()

    # Plot overall historgram
    hist_filename = "histogram.png"
    histogram_axis.hist(histogram_X, bins="auto", edgecolor="white")
    histogram_axis.set_xlabel("Length of history")
    histogram_axis.set_ylabel("Number of objects")
    historgram_fig.savefig(fname=os.path.join(dirpath, hist_filename), dpi='figure', format='png')
    plt.close()

def clustering_search_on_2D_feature_vectors(estimator, database: str, outdir: str, filter_threshold: float = (0.1, 0.7), n_jobs: int = -1, **estkwargs):
    # Load tack dataset
    trackedObjects = load_dataset(database)
    # Filter by yolov7 labels, ie. car, person, cycle
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    # Create 
    progress = 1
    thres_interval = 0.1
    max_progress = int(filter_threshold[1] / thres_interval)
    thres = filter_threshold[0]
    while thres <= filter_threshold[1]:
        filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
        if len(filteredTrackedObjects) > 0:
            clustering_on_2D_feature_vectors(
                estimator=estimator,
                trackedObjects=filteredTrackedObjects,
                outdir=outdir,
                n_jobs=n_jobs,
                filter_threshold=thres,
                **estkwargs
            )
        print(200 * '\n', '[', (progress-2) * '=', '>', int(max_progress-progress) * ' ', ']', flush=True)
        progress += 1
        print(f"Threshold value: {thres}")
        thres += thres_interval 

def clustering_search_on_4D_feature_vectors(estimator, database: str, outdir: str, filter_threshold: float = (0.1, 0.7), n_jobs: int = -1, **estkwargs):
    # Load tack dataset
    trackedObjects = load_dataset(database)
    # Filter by yolov7 labels, ie. car, person, cycle
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    # Create 
    progress = 1
    thres_interval = 0.1
    max_progress = int(filter_threshold[1] / thres_interval)
    thres = filter_threshold[0]
    while thres <= filter_threshold[1]:
        filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
        if len(filteredTrackedObjects) > 0:
            clustering_on_4D_feature_vectors(
                estimator=estimator,
                trackedObjects=filteredTrackedObjects,
                outdir=outdir,
                n_jobs=n_jobs,
                filter_threshold=thres,
                **estkwargs
            )
        print(200 * '\n', '[', (progress-2) * '=', '>', int(max_progress-progress) * ' ', ']', flush=True)
        progress += 1
        print(f"Threshold value: {thres}")
        thres += thres_interval 

def clustering_search_on_6D_feature_vectors(estimator, database: str, outdir: str, filter_threshold: float = (0.1, 0.7), n_jobs: int = -1, **estkwargs):
    # Load tack dataset
    trackedObjects = load_dataset(database)
    # Filter by yolov7 labels, ie. car, person, cycle
    trackedObjects = filter_tracks(trackedObjects) # filter out only cars
    # Create 
    progress = 1
    thres_interval = 0.1
    max_progress = int(filter_threshold[1] / thres_interval)
    thres = filter_threshold[0]
    while thres <= filter_threshold[1]:
        filteredTrackedObjects = filter_out_edge_detections(trackedObjects, thres)
        if len(filteredTrackedObjects) > 0:
            clustering_on_6D_feature_vectors(
                estimator=estimator,
                trackedObjects=filteredTrackedObjects,
                outdir=outdir,
                n_jobs=n_jobs,
                filter_threshold=thres,
                **estkwargs
            )
        print(200 * '\n', '[', (progress-2) * '=', '>', int(max_progress-progress) * ' ', ']', flush=True)
        progress += 1
        print(f"Threshold value: {thres}")
        thres += thres_interval 

def elbow_visualizer(X, k, model='kmeans', metric='silhouette', distance_metric='euclidean', show=False) -> plt.Figure:
    """Create elbow plot, to visualize what cluster number fits the best for the dataset.

    Args:
        X (np.ndarray): dataset for the clustering 
        k (int, tuple, list): number of clusters, can be an int, tuple, list,
                              if int then it will run clusters from 2 to given number 
        model (str, optional): Name of clustering algorithm. Defaults to 'kmeans'. Choices: 'kmeans', 'spectral'.
        metric (str, optional): The scoring metric, to score the clusterings with. Defaults to 'silhouette'. Choices: 'silhouette', 'calinksi-karabasz', 'davies-bouldin'.
        distance_metric (str, optional): Some of the metric algorithm need a distance metric. Defaults to 'euclidean'. For now this is the only one, but who knows what the future brings.
        show (bool): True if want to show plot

    Returns:
        plt.Figure: Returns a matplotlib figure object, that can be saved. 
    """
    if type(k) == int:
        n_clusters = [i for i in range(k)]
    elif type(k) == tuple:
        n_clusters = [i for i in range(k[0], k[1])]
    elif type(k) == list:
        n_clusters = [i for i in range(k[0], k[1])]
    if metric == 'silhouette':
        from sklearn.metrics import silhouette_score 
        if model == 'kmeans':
            cluster_labels = [k_means_on_featureVectors(X, n) for n in range(n_clusters[0], n_clusters[-1]+1)]
            scores = [silhouette_score(X, labels, metric=distance_metric) for labels in cluster_labels]
        elif model == 'spectral':
            cluster_labels = [spectral_on_featureVectors(X, n) for n in range(n_clusters[0], n_clusters[-1]+1)]
            scores = [silhouette_score(X, labels, metric=distance_metric) for labels in cluster_labels]
        elbow = scores.index(max(scores))
    elif metric == 'calinski-harabasz':
        from sklearn.metrics import calinski_harabasz_score
        if model == 'kmeans':
            cluster_labels = [k_means_on_featureVectors(X, n) for n in range(n_clusters[0], n_clusters[-1]+1)]
            scores = [calinski_harabasz_score(X, labels) for labels in cluster_labels]
        elif model == 'spectral':
            cluster_labels = [spectral_on_featureVectors(X, n) for n in range(n_clusters[0], n_clusters[-1]+1)]
            scores = [calinski_harabasz_score(X, labels) for labels in cluster_labels]
        elbow = scores.index(max(scores))
    elif metric == 'davies-bouldin':
        from sklearn.metrics import davies_bouldin_score 
        if model == 'kmeans':
            cluster_labels = [k_means_on_featureVectors(X, n) for n in range(n_clusters[0], n_clusters[-1]+1)]
            scores = [davies_bouldin_score(X, labels) for labels in cluster_labels]
        elif model == 'spectral':
            cluster_labels = [spectral_on_featureVectors(X, n) for n in range(n_clusters[0], n_clusters[-1]+1)]
            scores = [davies_bouldin_score(X, labels) for labels in cluster_labels]
        elbow = scores.index(min(scores))
    #score_diff = np.sign(np.diff(np.sign(np.diff(scores))))
    fig, ax = plt.subplots(1,1, figsize=(15,10))
    score_line = ax.plot(n_clusters, scores, marker='o', label=f'{metric} line')
    elbow_line = ax.axvline(n_clusters[elbow], ls='--', color='r', label=f'Elbow at k={n_clusters[elbow]},score={scores[elbow]}')
    #diff_line = ax[1].plot(n_clusters[:-2], score_diff, marker='o', ls='--', c=(0,1,0,0.2), label='differentiation line')
    ax.grid(True)
    #ax[1].grid(True)
    ax.set_title(f"{metric} score elbow for {model} clustering")
    ax.set_xlabel("k")
    ax.set_ylabel(f"{metric} score")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if show:
        plt.show()
    return fig

def elbow_on_clustering(X: np.ndarray, threshold: float, dirpath: str, model='kmeans', metric='silhouette', show=True):
    """Plot elbow diagram with kmeans and spectral clustering, and with different thresholds.
    Use this function instead of elbow_visualizer if want to save plot.

    Args:
        path2db (str): Path to database. 
        threshold (int, tuple, list): Give a range of threshold to do filtering with. 
        model (str): Name of cluster algorithm. Choices: 'silhouette', 'calinksi-harabasz', 'davies-bouldin'.
    """
    fig2save = elbow_visualizer(X, k=(2,16), model=model, metric=metric, show=show)
    filename = f"elbow_on_{model}_2-16_metric_{metric}_thresh_{threshold}.png"
    fig2save.savefig(fname=os.path.join(dirpath, filename))

def elbow_plot_worker(path2db: str, threshold=(0.1, 0.7), n_jobs=None):
    """This function generates mupltiple elbow plots, with different clustering algorithms, score metrics and thresholds.

    Args:
        path2db (str): Path to database file. 
        threshold (tuple, optional): Threshold range. Defaults to (0.01, 0.71).
    """
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    metrics = ['silhouette', 'calinski-harabasz', 'davies-bouldin']
    models = ['kmeans', 'spectral']
    dirpaths = {} 
    thres = threshold[0]
    # craete directory for elbow diagrams
    elbow_dirpath = os.path.join("research_data", path2db.split('/')[-1].split('.')[0], f"elbow_diagrams")
    # check if dir exists
    if not os.path.isdir(elbow_dirpath):
        # make dir if not
        os.mkdir(elbow_dirpath)
    for model in models:
        # create directory for models  
        model_dirpath= os.path.join(elbow_dirpath, f"{model}")
        # check if dir exists
        if not os.path.isdir(model_dirpath):
            # make dir if not
            os.mkdir(model_dirpath)
        dirpaths[model] = {}
        for metric in metrics:
            # create directory for metrics 
            dirpath = os.path.join(model_dirpath, f"{metric}")
            # check if dir exists
            if not os.path.isdir(dirpath):
                # make dir if not
                os.mkdir(dirpath)
            dirpaths[model][metric] = dirpath
    while thres < threshold[1]:
        filteredTracks = filter_out_false_positive_detections_by_enter_exit_distance(tracks, thres) 
        X = make_4D_feature_vectors(filteredTracks)
        for model in models:
            for metric in metrics:
                print(thres, model, metric)
                elbow_on_clustering(X, threshold=thres, dirpath=dirpaths[model][metric], model=model, metric=metric, show=False)
        thres += 0.1

def kmeans_mse_search(database: str, threshold: float = 0.7, n_jobs: int = 10, mse_threshold: float = 0.5, **estkwargs):
    from sklearn.cluster import KMeans
    from sklearn.cluster import OPTICS
    from visualizer import aoiextraction
    trackedObjects = load_dataset(database)
    trackedObjects = filter_out_edge_detections(trackedObjects, threshold)
    X = make_2D_feature_vectors(trackedObjects)
    _, labels = clustering_on_feature_vectors(X, OPTICS, n_jobs, **estkwargs)
    cluster_exitpoints = aoiextraction(trackedObjects, labels)
    mse = 9999
    n_clusters = 1
    n_clusters_best = n_clusters
    while(mse > mse_threshold):
        clr = KMeans(n_clusters, n_jobs=n_jobs).fit()
        aoi_labels = clr.labels_
        cluster_centers = clr.cluster_centers_
        print(aoi_labels)
        print(cluster_centers)

def elbow_plotter(path2db: str, threshold: float, model: str, metric: str, n_jobs=None):
    """Simply plots an elbow diagram with the given parameters.

    Args:
        path2db (str): Path to database file. 
        threshold (float): threshold value for filtering algorithm 
        model (str): clustering algorithm 
        metric (str): scoring metric 
    """
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    filteredTracks = filter_out_false_positive_detections(tracks, threshold)
    X = make_4D_feature_vectors(filteredTracks)
    dirpath = os.path.join("research_data", path2db.split('/')[-1].split('.')[0])
    elbow_on_clustering(X, threshold=threshold, dirpath=dirpath, model=model, metric=metric)

def elbow_on_kmeans(path2db: str, threshold: float, n_jobs=None):
    """Evaluate clustering results and create elbow diagram.

    Args:
        path2db (str): Path to database file. 
        threshold (float): Threshold value for filtering algorithm. 
    """
    from yellowbrick.cluster.elbow import kelbow_visualizer 
    from sklearn.cluster import KMeans
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs=n_jobs)
    filteredTracks = filter_out_false_positive_detections(tracks, threshold)
    X = make_4D_feature_vectors(filteredTracks)
    kelbow_visualizer(KMeans(), X, k=(2,10), metric='silhouette')
    kelbow_visualizer(KMeans(), X, k=(2,10), metric='calinski_harabasz')

def aoi_clutsering_search_birch(tracks_path, outdir, threshold, n_jobs=18, **estkwargs):
    from sklearn.cluster import Birch, OPTICS
    from processing_utils import load_joblib_tracks
    from visualizer import aoiextraction
    from scipy.stats import expon
    tracks = load_joblib_tracks(tracks_path)
    tracks_filtered = filter_out_edge_detections(trackedObjects=tracks, threshold=threshold)
    cls_samples = make_4D_feature_vectors(tracks_filtered)
    _, labels = clustering_on_feature_vectors(X=cls_samples, estimator=OPTICS, n_jobs=n_jobs, **estkwargs)
    tracks_labeled = tracks_filtered[labels > -1]
    cluster_labels = labels[labels > -1]

    cluster_aoi = aoiextraction(tracks_labeled, cluster_labels)

    thresholds = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    for th in tqdm.tqdm(thresholds):
        clr, aoi_labels = clustering_on_feature_vectors(cluster_aoi, Birch, n_jobs=n_jobs, threshold=th)
        # Generate dirname for plots
        dirpath = os.path.join(outdir, f"{clr}_n_clusters-{len(set(aoi_labels))}_threshold_{th}")
        # Create dir if it does not exists
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        reduced_labels = []
        for l in cluster_labels:
            reduced_labels.append(aoi_labels[l])
        for aoi_l in list(set(aoi_labels)):
            X = []
            Y = []
            fig, ax = plt.subplots(1,1, figsize=(15,8))
            for i, t in enumerate(tracks_labeled):
                if aoi_l == reduced_labels[i]:
                    ax.scatter([t.history_X[0]], [1-t.history_Y[0]], c='g')
                    ax.scatter([t.history_X[-1]], [1-t.history_Y[-1]], c='r')
                    for x,y in zip(t.history_X, t.history_Y):
                        X.append(x)
                        Y.append(y)
            ax.scatter(np.array(X), 1-np.array(Y), s=0.5)
            ax.grid(visible=True)
            ax.set_xlim(left=0.0, right=2.0)
            ax.set_ylim(bottom=0.0, top=2.0)
            ax.set_title(label=f"Threshold {th} cluster {aoi_l}")
            filename = os.path.join(dirpath, f"cluster_{aoi_l}")
            fig.savefig(filename)
            #plt.show()

def submodule_optics(args):
    #optics_worker(args.database, args.outdir, args.min_samples, args.xi, args.min_cluster_size, args.max_eps, n_jobs=args.n_jobs)
    from sklearn.cluster import OPTICS
    if args.dimensions == "2D":
        clustering_search_on_2D_feature_vectors(
            estimator=OPTICS, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            xi=args.xi, 
            min_cluster_size=args.min_cluster_size,
            p=args.p_norm
        )
    elif args.dimensions == "4D":
        clustering_search_on_4D_feature_vectors(
            estimator=OPTICS, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            xi=args.xi, 
            min_cluster_size=args.min_cluster_size,
            p=args.p_norm
        )
    elif args.dimensions == "6D":
        clustering_search_on_6D_feature_vectors(
            estimator=OPTICS, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            xi=args.xi, 
            min_cluster_size=args.min_cluster_size,
            p=args.p_norm
        )

def submodule_birch(args):
    from sklearn.cluster import Birch 
    if args.dimensions == "2D":
        clustering_search_on_2D_feature_vectors(
            estimator=Birch, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            threshold=args.threshold,
            branching_factor=args.branching,
            n_clusters=args.n_clusters
        )
    elif args.dimensions == "4D":
        clustering_search_on_4D_feature_vectors(
            estimator=Birch, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            threshold=args.threshold,
            branching_factor=args.branching,
            n_clusters=args.n_clusters
        )
    elif args.dimensions == "6D":
        clustering_search_on_6D_feature_vectors(
            estimator=Birch, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            threshold=args.threshold,
            branching_factor=args.branching,
            n_clusters=args.n_clusters
        )

def submodule_kmeans(args):
    from sklearn.cluster import KMeans 
    if args.dimensions == "4D":
        clustering_search_on_4D_feature_vectors(
            estimator=KMeans, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            n_clusters=args.n_clusters
        )
    if args.dimensions == "6D":
        clustering_search_on_6D_feature_vectors(
            estimator=KMeans, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            n_clusters=args.n_clusters
        )

def submodule_dbscan(args):
    from sklearn.cluster import DBSCAN 
    if args.dimensions == "4D":
        clustering_search_on_4D_feature_vectors(
            estimator=DBSCAN, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            eps=args.eps,
            min_samples=args.min_samples,
            p=args.p_norm
        )
    if args.dimensions == "6D":
        clustering_search_on_6D_feature_vectors(
            estimator=DBSCAN, 
            database=args.database, 
            outdir=args.outdir,
            n_jobs=args.n_jobs,
            eps=args.eps,
            min_samples=args.min_samples,
            p=args.p_norm
        )

def submodule_aoi_birch(args):
    aoi_clutsering_search_birch(args.database, 
        args.outdir,
        args.threshold, 
        n_jobs=args.n_jobs, 
        min_samples=args.min_samples, 
        max_eps=args.max_eps, 
        xi=args.xi
    )

def submodule_aoi_kmeans(args):
    kmeans_mse_search(args.database,
                      args.threshold,
                      args.n_jobs,
                      args.mse,
                      args.min_samples,
                      args.max_eps,
                      args.xi,
                      args.pnorm
    )

def main():
    import argparse
    argparser = argparse.ArgumentParser("Analyze results of main program. Make and save plots. Create heatmap or use clustering on data stored in the database.")
    argparser.add_argument("-db", "--database", help="Path to joblib dataset.", required=True)
    argparser.add_argument("--outdir", "-o", help="Output directory path.", required=True)
    argparser.add_argument("--dimensions", type=str, choices=["2D", "4D", "6D"], help="Choose the dimensions of the feature vector.", required=True)
    argparser.add_argument("--n_jobs", type=int, help="Number of processes.", default=None)

    subparser = argparser.add_subparsers(help="Chose from clustering methods.")

    optics_parser = subparser.add_parser("optics", help="OPTICS clustering.")
    optics_parser.add_argument("--min_samples", default=10, type=int, help="Set minimum sample number for a cluster.")
    optics_parser.add_argument("--max_eps", type=float, default=np.inf, help="Set maximum epsilon distance that can be between samples of a cluster.")
    optics_parser.add_argument("--xi", type=float, default=0.15, help="Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.")
    optics_parser.add_argument("--min_cluster_size", type=float, default=None, help="Minimum number of samples in an OPTICS cluster, expressed as an absolute number or" 
                                                                              "a fraction of the number of samples (rounded to be at least 2). If flag not used," 
                                                                              "then min_cluster_size = max_samples.")
    optics_parser.add_argument("-p", "--p_norm", type=int, default=2, help="Set p norm parameter of OPTICS clustering, to affect metrics.")
    optics_parser.set_defaults(func=submodule_optics)

    birch_parser = subparser.add_parser("birch", help="Birch clustering.")
    birch_parser.add_argument("--threshold", type=float, default=0.5, help="The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold." 
                                                                           "Otherwise a new subcluster is started. Setting this value to be very low promotes splitting and vice-versa.")
    birch_parser.add_argument("--branching", type=int, default=50,
        help="Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then that node is split into two nodes" 
             "with the subclusters redistributed in each. The parent subcluster of that node is removed and two new subclusters are added as parents of the 2 split nodes."
    )
    birch_parser.add_argument("--n_clusters", type=int, default=None, 
        help="Number of clusters after the final clustering step, which treats the subclusters from the leaves as new samples."
    )
    birch_parser.set_defaults(func=submodule_birch)

    kmeans_parser = subparser.add_parser("kmeans", help="KMeans clustering.")
    kmeans_parser.add_argument("--n_clusters", type=int, default=10, help="Number of clusters.")
    kmeans_parser.set_defaults(func=submodule_kmeans)

    dbscan_parser = subparser.add_parser("dbscan", help="DBSCAN clustering.")
    dbscan_parser.add_argument("--min_samples", default=10, type=int, help="Set minimum sample number for a cluster.")
    dbscan_parser.add_argument("--eps", type=float, default=np.inf, help="Set epsilon distance that can be between samples of a cluster.")
    dbscan_parser.add_argument("-p", "--p_norm", type=int, default=2, help="Set p norm parameter of OPTICS clustering, to affect metrics.")
    dbscan_parser.set_defaults(func=submodule_dbscan)

    aoi_optics = subparser.add_parser("aoi_optics_search", help="Cluster reduction estimator parameter search.")
    aoi_optics.add_argument("--min_samples", default=10, type=int, help="Set minimum sample number for a cluster.")
    aoi_optics.add_argument("--max_eps", type=float, default=np.inf, help="Set maximum epsilon distance that can be between samples of a cluster.")
    aoi_optics.add_argument("--xi", type=float, default=0.15, help="Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.")
    aoi_optics.add_argument("--threshold", type=float, default=0.7)
    aoi_optics.set_defaults(func=submodule_aoi_birch)

    kmeans_mse_search_parser = subparser.add_parser("kmeans_mse_search", help="KMeans clustering with max squared error search.")
    kmeans_mse_search_parser.add_argument("--mse", type=float, default=0.5,
                                          help="Mean squared error threshold.")
    kmeans_mse_search_parser.add_argument("--")

    args = argparser.parse_args() 
    args.func(args)

    """
    argparser.add_argument("--kmeans", help="Use kmeans flag to run kmeans clustering on detection data.", action="store_true", default=False)
    argparser.add_argument("--kmeans_batch_plot", help="Run batch plotter on kmeans clustering.", action="store_true", default=False)
    argparser.add_argument("--n_clusters", type=int, default=2, help="KMEANS, SPECTRAL parameter: number of clusters to make.")
    argparser.add_argument("--threshold", type=float, default=0.5, help="Threshold value for filtering algorithm that filters out the best detections.")
    argparser.add_argument("--spectral", help="Use spectral flag to run spectral clustering on detection data.", action="store_true", default=False)
    argparser.add_argument("--spectral_batch_plot", help="Run batch plotter on spectral clustering.", action="store_true", default=False)
    argparser.add_argument("--affinity_on_enters_and_exits", help="Use this flag to run affinity propagation clustering on extracted feature vectors.", default=False, action="store_true")
    argparser.add_argument("--elbow_on_kmeans", type=str, choices=['silhouette', 'calinski-harabasz', 'davies-bouldin'], help="Choose which metric to score kmeans clustering.")
    argparser.add_argument("--elbow_on_spectral", type=str, choices=['silhouette', 'calinski-harabasz', 'davies-bouldin'], help="Choose which metric to score kmeans clustering.")
    argparser.add_argument("--plot_elbows", action='store_true', help="This function helps to plot all kinds of elbow diagrams and save them.")
    argparser.add_argument("--dbscan_batch_plot", help="Run batch plotter on dbscan clustering.", default=False, action="store_true")
    argparser.add_argument("--eps", default=0.1, type=float, help="DBSCAN and OPTICS_DBSCAN parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    argparser.add_argument("--shuffle_dataset", default=False, action="store_true", help="DBSCAN parameter: Shuffle dataset for slightly different clustering results.")
    argparser.add_argument("--optics_batch_plot", help="Run batch plotter on optics clustering.", action="store_true", default=False)
    
    argparser.add_argument("--cluster_optics_dbscan_batch_plot", help="Run batch plot on optics and dbscan hybrid.", default=False, action="store_true")
    args = argparser.parse_args()

    if args.kmeans and args.threshold and args.n_clusters and args.outdir:
        simple_kmeans_plotter(args.database, args.outdir, args.threshold, args.n_clusters, args.n_jobs)
    if args.kmeans_batch_plot and args.outdir and args.database:
        kmeans_worker(args.database, args.outdir, n_jobs=args.n_jobs)
    if args.spectral_batch_plot:
        spectral_worker(args.database, args.outdir, n_jobs=args.n_jobs)
    if args.spectral and args.threshold and args.n_clusters:
        simple_spectral_plotter(args.database, args.outdir, args.threshold, args.n_clusters, args.n_jobs)
    if args.dbscan_batch_plot:
        dbscan_worker(args.database, eps=args.eps, min_samples=args.min_samples, n_jobs=args.n_jobs, shuffle=args.shuffle_dataset)
    if args.optics_batch_plot:
        if args.min_cluster_size:
            optics_worker(args.database, args.outdir, args.min_samples, args.xi, args.min_cluster_size, args.max_eps, n_jobs=args.n_jobs)
        else:
            optics_worker(args.database, args.outdir, args.min_samples, args.xi, args.min_samples, args.max_eps, n_jobs=args.n_jobs)
    if args.cluster_optics_dbscan_batch_plot:
        if args.min_cluster_size:
            optics_dbscan_worker(args.database, args.outdir, args.min_samples, args.xi, args.min_cluster_size, args.eps, n_jobs=args.n_jobs)
        else:
            optics_dbscan_worker(args.database, args.outdir, args.min_samples, args.xi, args.min_samples, args.eps, n_jobs=args.n_jobs)
    if args.elbow_on_kmeans:
        elbow_plotter(args.database, args.threshold, model='kmeans', metric=args.elbow_on_kmeans, n_jobs=args.n_jobs)
    if args.elbow_on_spectral:
        elbow_plotter(args.database, args.threshold, model='spectral', metric=args.elbow_on_spectral, n_jobs=args.n_jobs)
    if args.plot_elbows:
        elbow_plot_worker(args.database, n_jobs=args.n_jobs)
    """

if __name__ == "__main__":
    main()