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
import databaseLoader
from dataManagementClasses import Detection, TrackedObject
import numpy as np
import matplotlib.pyplot as plt
import os

def detectionFactory(objID: int, frameNum: int, label: str, confidence: float, x: float, y: float, width: float, height: float, vx: float, vy: float, ax:float, ay: float) -> Detection:
    """Creates Detection object.

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

def cvCoord2npCoord(Y: np.ndarray) -> np.ndarray:
    """Convert OpenCV Y axis coordinates to numpy coordinates.

    Args:
        Y (np.ndarray): Y axis coordinate vector

    Returns:
        np.ndarray: Y axis coordinate vector
    """
    return 1 - Y

def detectionParser(rawDetectionData):
    detections = []
    for entry in rawDetectionData:
        detections.append(detectionFactory(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7], entry[8], entry[9], entry[10], entry[11]))
    return detections 

def makeColormap(path2db):
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

def kmeans_clustering(path2db):
    from sklearn.cluster import KMeans 
    from itertools import cycle
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    x = np.array([det.X for det in detections])
    y = np.array([det.Y for det in detections])
    y = cvCoord2npCoord(y)
    X = np.array([[x,y] for x,y in zip(x,y)]) 
    cluster = KMeans(n_clusters=2).fit(X)
    labels = cluster.labels_
    x_0 = np.array([x[i] for i in range(0, len(x)) if labels[i] == 0])
    y_0 = np.array([y[i] for i in range(0, len(y)) if labels[i] == 0])
    x_1 = np.array([x[i] for i in range(0, len(x)) if labels[i] == 1])
    y_1 = np.array([y[i] for i in range(0, len(y)) if labels[i] == 1])
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    print(f"Size of x_0: {len(x_0)} \n Size of x_1: {len(x_1)} \n")
    fig, axes = plt.subplots()
    axes.scatter(x_0, y_0, c='r', s=0.3)
    axes.scatter(x_1, y_1, c='g', s=0.3)
    fig2save = plt.gcf()
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_kmeans_n_cluster_2"
    fig2save.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

def findDirections(path2db):
    rawDetData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetData)
    print(detections[::10])
    

def main():
    argparser = argparse.ArgumentParser("Create plots from database data.")
    argparser.add_argument("-db", "--database", help="Path to database file.")
    argparser.add_argument("-hm", "--heatmap", help="Use this flag if want to make a heatmap from the database data.", action="store_true", default=False)
    argparser.add_argument("-c", "--config", help="Print configuration used for the video.", action="store_true", default=False)
    argparser.add_argument("-cl", "--cluster", help="Use this flag to create clusters from video data.", action="store_true", default=False)
    argparser.add_argument("-fd", "--findDirections", action="store_true", default=False, help="Use this flag, when want to find directions.")
    args = argparser.parse_args()
    if args.config:
        printConfig(args.database)
    if args.heatmap:
        if not os.path.isdir(os.path.join("research_data", args.database.split('/')[-1].split('.')[0])):
            os.mkdir(os.path.join("research_data", args.database.split('/')[-1].split('.')[0]))
            print("Directory \"research_data/{}\" is created.".format(args.database.split('/')[-1].split('.')[0]))
        coordinates2heatmap(args.database)
    if args.cluster:
        if not os.path.isdir(os.path.join("research_data", args.database.split('/')[-1].split('.')[0])):
            os.mkdir(os.path.join("research_data", args.database.split('/')[-1].split('.')[0]))
            print("Directory \"research_data/{}\" is created.".format(args.database.split('/')[-1].split('.')[0]))
        kmeans_clustering(args.database)

if __name__ == "__main__":
    main()
