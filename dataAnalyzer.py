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
import argparse
import time
import databaseLoader
from dataManagementClasses import Detection, TrackedObject
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

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

def coordinates2heatmap(path2db):
    """Create heatmap from detection data.
    Every object has its own coloring.

    Args:
        path2db (str): Path to database file. 
    """
    databaseDetections = databaseLoader.loadDetections(path2db)
    detections = detectionParser(databaseDetections) 
    X = np.array([det.X for det in detections])
    Y = np.array([det.Y for det in detections])
    # converting Y coordinates, becouse in opencv, coordinates start from top to bottom, ex.: coordinate (0,0) is in top left corner, not bottom left
    Y = cvCoord2npCoord(Y) 
    fig, ax1 = plt.subplots(1,1)
    colormap = makeColormap(path2db)
    ax1.scatter(X, Y, np.ones_like(X), colormap)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_heatmap"
    fig.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

def printConfig(path2db):
    metadata = databaseLoader.loadMetadata(path2db)
    regression = databaseLoader.loadRegression(path2db)
    historydepth = metadata[0][0]
    futuredepth = metadata[0][1]
    yoloVersion = metadata[0][2]
    device = metadata[0][3]
    imgsize = metadata[0][4]
    stride = metadata[0][5]
    confThres = metadata[0][6]
    iouThres = metadata[0][7]
    linearFunc = regression[0][0]
    polynomFunc = regression[0][1]
    polynomDegree = regression[0][2]
    trainingPoints = regression[0][3]
    print(
    f"""
        Yolo config:
            Version: {yoloVersion}
            Device: {device}
            NN image size: {imgsize}
            Confidence threshold: {confThres}
            IOU threshold: {iouThres}
            Convolutional kernel stride: {stride}
        Regression config:
            Length of detection history: {historydepth}
            Number of predicted coordinates: {futuredepth}
            Number of detections used for training: {trainingPoints}
            Name of function used in linear regression: {linearFunc}
            Name of function used in polynom regression: {polynomFunc}
            Degree of polynoms used for the regression: {polynomDegree}
    """
    )

# TODO: implement multiprocessed version of this function 
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

def findEnterAndExitPointsMultithreaded(path2db: str):
    """Extract only the first and last detections of tracked objects.
    This is a multithreaded implementation of function findEnterAndExitPoints(path2db: str)

    Args:
        path2db (str): Path to database file. 
    """
    from multiprocessing import Pool
    import concurrent.futures
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    rawObjectData = databaseLoader.loadObjects(path2db)
    enterDetections = []
    exitDetections = []
    iterable = [[detections, obj[0]] for obj in rawObjectData]
    with Pool(processes=20) as executor:
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
    print(filteredTracks)
    return filteredTracks


def makeFeatureVectors_Nx2(detections: list, ) -> np.ndarray:
    """Create feature vectors from inputted detections.

    Args:
        detections (list): Any list that contains detecion objects. 

    Returns:
        np.ndarray: numpy ndarray of shape ({length of detections}, 2) 
    """
    # create ndarray of ndarrays containing the x,y coordinated of detections
    featureVectors = np.array([np.array([det.X, det.Y]) for det in detections])
    return featureVectors

def makeFeatureVectorsNx4(trackedObjects: list) -> np.ndarray:
    """Create feature vectors from the two inputted detection lists.
    The vector is created from the detections_a x,y and the detections_b x,y coordinates.

    Args:
        detections_a (list): detection list a 
        detections_b (list): detection list b 

    Returns:
        np.ndarray: 
    """
    featureVectors = np.array([np.array([obj.history[0].X, obj.history[0].Y, obj.history[-1].X, obj.history[-1].Y]) for obj in tqdm.tqdm(trackedObjects, desc="Feature vectors.")])
    return featureVectors

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
    filteredTrackedObjects = filter_out_false_positive_detections(trackedObjects, threshold)
    featureVectors = makeFeatureVectorsNx4(filteredTrackedObjects)
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
    

def kmeans_clustering_on_nx2(path2db: str, n_clusters: int, threshold: float):
    """Run kmeans clustering on filtered feature vectors.

    Args:
        path2db (str): Path to database file. 
        n_clusters (int): number of initial clusters for kmeans 
        threshold (float): the threshold for the filtering algorithm 
    """
    filteredEnterDets, filteredExitDets = filter_out_false_positive_detections(path2db, threshold)
    filteredEnterFeatures  = makeFeatureVectors(filteredEnterDets)
    filteredExitFeatures = makeFeatureVectors(filteredExitDets)
    colors = "bgrcmyk"
    labels_enter = k_means_on_featureVectors(filteredEnterFeatures, n_clusters)
    labels_exit = k_means_on_featureVectors(filteredExitFeatures, n_clusters)
    fig, axes = plt.subplots(n_clusters,1, figsize=(10,10))
    axes[0].set_xlim(0,1)
    axes[0].set_ylim(0,1)
    axes[1].set_xlim(0,1)
    axes[1].set_ylim(0,1)
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

def kmeans_clustering_on_nx4(trackedObjects: list, n_clusters: int, threshold: float, path2db: str):
    """Run kmeans clutering on N x 4 (x,y,x,y) feature vectors.

    Args:
        trackedObjects (list): List of object tracks. 
        n_clusters (int): Number of clusters. 
        threshold (float): Threshold value for the false positive filter algorithm. 
    """
    featureVectors = makeFeatureVectorsNx4(trackedObjects)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    labels = k_means_on_featureVectors(featureVectors, n_clusters)
    fig, axes = plt.subplots(n_clusters,1, figsize=(10,10))
    if n_clusters > 1:
        for i in range(n_clusters):
            axes[i].set_xlim(0,1)
            axes[i].set_ylim(0,1)   
            axes[i].set_title(f"Axis of cluster number {i}")
            enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==i])
            enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes[i].scatter(enter_x, enter_y, c='g', s=6, label=f"Enter points")
            exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==i])
            exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==i])
            axes[i].scatter(exit_x, exit_y, c='r', s=6, label=f"Exit points")
            axes[i].legend()
    else:
        axes.set_xlim(0,1)
        axes.set_ylim(0,1)   
        axes.set_title(f"Axis of cluster number {0}")
        enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==0])
        enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==0])
        axes.scatter(enter_x, enter_y, c='g', s=6, label=f"Enter points")
        exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==0])
        exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==0])
        axes.scatter(exit_x, exit_y, c='r', s=6, label=f"Exit points")
        axes.legend()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_kmeans_on_nx4_n_cluster_{n_clusters}_threshold_{threshold}_dets_{len(featureVectors)}.png"
    fig.savefig(fname=os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi='figure', format='png')

def kmeans_worker(path2db: str, threshold: float, k=(4,5)):
    """This function automates the task of running kmeans clustering on different cluster numbers.

    Args:
        path2db (str): path to database file 
        n_cluster_start (int): starting number cluster 
        n_cluster_end (int): ending number cluster 
        threshold (float): threshold for filtering algorithm 

    Returns:
        bool: returns false if some crazy person uses the program 
    """
    if k[0] < 1 or k[1] < k[0]:
        print("Error: this is not how we use this program properly")
        return False
    rawObjectData = databaseLoader.loadObjects(path2db)
    trackedObjects = []
    for rawObj in tqdm.tqdm(rawObjectData, desc="Loading detections of tracks."):
        tmpDets = []
        rawDets = databaseLoader.loadDetectionsOfObject(path2db, rawObj[0])
        if len(rawDets) > 0:
            tmpDets = detectionParser(rawDets)
            trackedObjects.append(trackedObjectFactory(tmpDets))
    filteredTrackedObjects = filter_out_false_positive_detections(trackedObjects, threshold)
    for i in range(k[0], k[1]):
        kmeans_clustering_on_nx4(filteredTrackedObjects, i, threshold, path2db)

def spectral_clustering_on_nx4(path2db: str, n_clusters: int, threshold: float):
    """Run spectral clustering on N x 4 (x,y,x,y) feature vectors.

    Args:
        path2db (str): Path to database file. 
        n_clusters (int): Number of clusters. 
        threshold (float): Threshold value for the false positive filter algorithm. 
    """
    from sklearn.cluster import SpectralClustering 
    filteredEnterDets, filteredExitDets = filter_out_false_positive_detections(path2db, threshold)
    featureVectors = makeFeatureVectorsNx4(filteredEnterDets, filteredExitDets)
    print(f"Number of feature vectors: {len(featureVectors)}")
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    spec = SpectralClustering(n_clusters=n_clusters, n_jobs=-1).fit(featureVectors)
    labels = spec.labels_ 
    fig, axes = plt.subplots(2,1, figsize=(10,10))
    axes[0].set_xlim(0,1)
    axes[0].set_ylim(0,1)
    axes[1].set_xlim(0,1)
    axes[1].set_ylim(0,1)
    axes[0].set_title("Clusters of enter points")
    axes[1].set_title("Clusters of exit points")
    for i in range(n_clusters):
        enter_x = np.array([featureVectors[idx][0] for idx in range(len(featureVectors)) if labels[idx]==i])
        enter_y = np.array([1-featureVectors[idx][1] for idx in range(len(featureVectors)) if labels[idx]==i])
        axes[0].scatter(enter_x, enter_y, c=colors[i], s=6, label=f"Cluster of number {i}")
        exit_x = np.array([featureVectors[idx][2] for idx in range(len(featureVectors)) if labels[idx]==i])
        exit_y = np.array([1-featureVectors[idx][3] for idx in range(len(featureVectors)) if labels[idx]==i])
        axes[1].scatter(exit_x, exit_y, c=colors[i], s=6, label=f"Cluster of number {i}")
    axes[0].legend()
    axes[1].legend()
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_spectral_on_nx4_n_cluster_{n_clusters}_threshold_{threshold}.png"
    fig.savefig(fname=os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi='figure', format='png')

def checkDir(path2db):
    """Check for dir of given database, to be able to save plots.

    Args:
        path2db (str): Path to database. 
    """
    if not os.path.isdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0])):
        os.mkdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0]))
        print("Directory \"research_data/{}\" is created.".format(path2db.split('/')[-1].split('.')[0]))

def elbow_on_kmeans(path2db: str, threshold: float):
    """Evaluate clustering results and create elbow diagram.

    Args:
        path2db (str): Path to database file. 
        threshold (float): Threshold value for filtering algorithm. 
    """
    from yellowbrick.cluster.elbow import kelbow_visualizer 
    from sklearn.cluster import KMeans
    filteredEnterDets, filteredExitDets = filter_out_false_positive_detections(path2db, threshold)
    X = makeFeatureVectorsNx4(filteredEnterDets, filteredExitDets)
    kelbow_visualizer(KMeans(), X, k=(2,10))

def main():
    argparser = argparse.ArgumentParser("Analyze results of main program. Make and save plots. Create heatmap or use clustering on data stored in the database.")
    argparser.add_argument("-db", "--database", help="Path to database file.")
    argparser.add_argument("-hm", "--heatmap", help="Use this flag if want to make a heatmap from the database data.", action="store_true", default=False)
    argparser.add_argument("-c", "--config", help="Print configuration used for the video.", action="store_true", default=False)
    argparser.add_argument("--kmeans", help="Use kmeans flag to run kmeans clustering on detection data.", action="store_true", default=False)
    argparser.add_argument("--n_clusters", type=int, default=2, help="If kmeans, spectral is chosen, set number of clusters.")
    argparser.add_argument("--threshold", type=float, default=0.01, help="When kmean and spectral clustering flag used, use this flag to give a threshold value to filter out false positive detections.")
    #argparser.add_argument("--spectral", help="Use spectral flag to run spectral clustering on detection data.", action="store_true", default=False)
    argparser.add_argument("--affinity_on_enters_and_exits", help="Use this flag to run affinity propagation clustering on extracted feature vectors.", default=False, action="store_true")
    argparser.add_argument("--elbow_on_kmeans", default=False, action="store_true", help="Use this flag to draw elbow diagram, from kmeans clustering results.")
    #argparser.add_argument("--filter_enter_and_exit", help="Use this flag when want to visualize objects that enter and exit point distance were lower than the given threshold. Threshold must be between 0 and 1.", default="0.01", type=float)
    args = argparser.parse_args()
    if args.database is not None:
        checkDir(args.database)
    if args.config:
        printConfig(args.database)
    if args.heatmap:
        coordinates2heatmap(args.database)
    if args.kmeans:
        # start cluster number is still hardcoded
        kmeans_worker(args.database, args.threshold, (2, args.n_clusters+1))
    #if args.spectral:
    #    spectral_clustering_on_nx4(args.database, args.n_clusters, args.threshold)
    if args.affinity_on_enters_and_exits:
        affinityPropagation_on_enter_and_exit_points(args.database, args.threshold)
    if args.elbow_on_kmeans:
        elbow_on_kmeans(args.database, args.threshold)
    
if __name__ == "__main__":
    main()
