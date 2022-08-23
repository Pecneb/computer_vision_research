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
    fig2save = plt.gcf()
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_heatmap"
    fig2save.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

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

def kmeans_clustering(path2db, n_clusters):
    """Use kmean algorithm to cluster detection data.
    The number of clusters have to be given initially for k_means algorithm.

    Args:
        path2db (str): Path to the datbase file. 
        n_clusters (int): Number of clusters in data.
    """
    from sklearn.cluster import KMeans 
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    x = np.array([det.X for det in detections])
    y = np.array([det.Y for det in detections])
    y = cvCoord2npCoord(y)
    X = np.array([[x,y] for x,y in zip(x,y)]) 
    # Number of clusters have to be given
    cluster = KMeans(n_clusters=n_clusters).fit(X)
    # A list with the cluster numberings. Same lenght as the X.
    labels = cluster.labels_
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    # print(f"Size of x_0: {len(x_0)} \n Size of x_1: {len(x_1)} \n")
    fig, axes = plt.subplots()
    # Using scatter plot to show the clusters with different coloring.
    for idx in range(n_clusters):
        # Exctracting cluster number {idx} from vector x and y
        _x= np.array([x[i] for i in range(0, len(x)) if labels[i] == idx])
        _y = np.array([y[i] for i in range(len(y)) if labels[i] == idx])
        axes.scatter(_x, _y, c=colors[idx], s=0.3)
    fig2save = plt.gcf()
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_kmeans_n_cluster_{n_clusters}"
    fig2save.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

def spectral_clustering(path2db, n_clusters):
    from sklearn.cluster import SpectralClustering 
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    x = np.array([det.X for det in detections])
    y = np.array([det.Y for det in detections])
    y = cvCoord2npCoord(y)
    X = np.array([[x,y] for x,y in zip(x,y)])
    labels = SpectralClustering(n_clusters=n_clusters).fit_predict(X)
    colors = "bgrcmykbgrcmykbgrcmykbgrcmyk"
    # print(f"Size of x_0: {len(x_0)} \n Size of x_1: {len(x_1)} \n")
    fig, axes = plt.subplots()
    # Using scatter plot to show the clusters with different coloring.
    for idx in range(n_clusters):
        # Exctracting cluster number {idx} from vector x and y
        _x= np.array([x[i] for i in range(0, len(x)) if labels[i] == idx])
        _y = np.array([y[i] for i in range(len(y)) if labels[i] == idx])
        axes.scatter(_x, _y, c=colors[idx], s=0.3)
    fig2save = plt.gcf()
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_spectral_n_cluster_{n_clusters}"
    fig2save.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

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
    for obj in rawObjectData:
        tmpDets = []
        for det in detections:
           if det.objID == obj[0]:
            tmpDets.append(det)
        if len(tmpDets) > 0:
            trackedObjects.append(trackedObjectFactory(tmpDets))
    enterDetections = [obj.history[0] for obj in trackedObjects]
    exitDetections = [obj.history[-1] for obj in trackedObjects]
    return enterDetections, exitDetections 

def makeFeatureVectors(detections: list) -> np.ndarray:
    """Create feature vectors from inputted detections.

    Args:
        detections (list): Any list that contains detecion objects. 

    Returns:
        np.ndarray: numpy ndarray of shape ({length of detections}, 2) 
    """
    # create ndarray of ndarrays containing the x,y coordinated of detections
    featureVectors = np.array([np.array([det.X, det.Y]) for det in detections])
    return featureVectors

def affinityPropagation_on_featureVector(featureVectors, path2db):
    """Run affinity propagation clustering algorithm on list of feature vectors. 

    Args:
        featureVector (list): A numpy ndarray of numpy ndarrays. ex.: [[x,y], [x2,y2]] 
    """
    from sklearn.cluster import AffinityPropagation 
    from itertools import cycle
    af= AffinityPropagation(random_state=5).fit(featureVectors)
    cluster_center_indices_= af.cluster_centers_indices_
    labels_ = af.labels_ 
    n_clusters_= len(cluster_center_indices_)
    print("Estimated number of clusters: %d" % n_clusters_)
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    fig, axes = plt.subplots()
    for k, col in zip(range(n_clusters_), colors):
        axes.scatter(np.array([featureVectors[idx, 0] for idx in range(len(labels_)) if labels_[idx]==k]), 
        np.array([1-featureVectors[idx, 1] for idx in range(len(labels_)) if labels_[idx]==k]), c=col)
    fig2save = plt.gcf()
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_affinity_propagation_featureVectors_n_clusters_{n_clusters_}_{time.strftime('%Y%m%d_%H:%M:%S')}"
    fig2save.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

def affinityPropagation_on_enter_and_exit_points(path2db):
    enter, exit = findEnterAndExitPoints(path2db)
    enterFeatures= makeFeatureVectors(enter)
    exitFeatures= makeFeatureVectors(exit)
    affinityPropagation_on_featureVector(enterFeatures, path2db)
    affinityPropagation_on_featureVector(exitFeatures, path2db)

def checkDir(path2db):
    """Check for dir of given database, to be able to save plots.

    Args:
        path2db (str): Path to database. 
    """
    if not os.path.isdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0])):
        os.mkdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0]))
        print("Directory \"research_data/{}\" is created.".format(path2db.split('/')[-1].split('.')[0]))

def main():
    argparser = argparse.ArgumentParser("Analyze results of main program. Make and save plots. Create heatmap or use clustering on data stored in the database.")
    argparser.add_argument("-db", "--database", help="Path to database file.")
    argparser.add_argument("-hm", "--heatmap", help="Use this flag if want to make a heatmap from the database data.", action="store_true", default=False)
    argparser.add_argument("-c", "--config", help="Print configuration used for the video.", action="store_true", default=False)
    argparser.add_argument("--kmeans", help="Use kmeans flag to run kmeans clustering on detection data.", action="store_true", default=False)
    argparser.add_argument("--n_clusters", type=int, default=2, help="If kmeans, spectral is chosen, set number of clusters.")
    argparser.add_argument("-fd", "--findDirections", action="store_true", default=False, help="Use this flag, when want to find directions.")
    argparser.add_argument("--spectral", help="Use spectral flag to run spectral clustering on detection data.", action="store_true", default=False)
    argparser.add_argument("--affinity_on_enters_and_exits", help="Use this flag to run affinity propagation clustering on extracted feature vectors.", default=False, action="store_true")
    args = argparser.parse_args()
    if args.database is not None:
        checkDir(args.database)
    if args.config:
        printConfig(args.database)
    if args.heatmap:
        coordinates2heatmap(args.database)
    if args.kmeans:
        kmeans_clustering(args.database, args.n_clusters)
    if args.spectral:
        spectral_clustering(args.database, args.n_clusters)
    if args.affinity_on_enters_and_exits:
        affinityPropagation_on_enter_and_exit_points(args.database)


if __name__ == "__main__":
    main()
