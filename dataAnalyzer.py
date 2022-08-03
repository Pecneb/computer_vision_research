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


def coordinates2heatmap(path2db):
    databaseDetections = databaseLoader.loadDetections(path2db)
    detections = []
    for entry in databaseDetections:
        detections.append(detectionFactory(entry[0], entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7], entry[8], entry[9], entry[10], entry[11]))
    X = np.array([det.X for det in detections])
    Y = np.array([det.Y for det in detections])
    fig, ax1 = plt.subplots(1,1)
    ax1.scatter(X, Y)
    ax1.set_xlim(0,2)
    ax1.set_ylim(0,2)
    plt.show()

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

def main():
    argparser = argparse.ArgumentParser("Create plots from database data.")
    argparser.add_argument("-db", "--database", help="Path to database file.")
    argparser.add_argument("-hm", "--heatmap", help="Use this flag if want to make a heatmap from the database data.", action="store_true", default=False)
    argparser.add_argument("-c", "--config", help="Print configuration used for the video.", action="store_true", default=False)
    args = argparser.parse_args()
    if args.heatmap:
        coordinates2heatmap(args.database)
    if args.config:
        printConfig(args.database)

if __name__ == "__main__":
    main()
