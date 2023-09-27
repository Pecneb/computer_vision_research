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

### Third Part ###
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

### Local ###
import utility.databaseLoader as databaseLoader 
from utility.general import (
    checkDir
)
from utility.plots import (
    cvCoord2npCoord,
)
from utility.dataset import (
    load_dataset
)

# disable sklearn warning
def warn(*arg, **args):
    pass
import warnings
warnings.warn = warn

def coordinates2heatmap(path2db):
    """Create heatmap from detection data.
    Every object has its own coloring.

    Args:
        path2db (str): Path to database file. 
    """
    path = Path(path2db)
    if path.is_dir():
        tracks = np.array([])
        for p in path.glob("*.joblib"):
            tracks = np.append(tracks, load_dataset(p))
            print(len(tracks))
    else:
        tracks = load_dataset(path)
    print(len(tracks))
    X = np.array([det.X for t in tracks for det in t.history])
    Y = np.array([det.Y for t in tracks for det in t.history])
    # converting Y coordinates, because in opencv, coordinates start from top to bottom, ex.: coordinate (0,0) is in top left corner, not bottom left
    Y = cvCoord2npCoord(Y) 
    fig, ax1 = plt.subplots(1,1)
    #colormap = makeColormap(path2db)
    ax1.scatter(X, Y, s=0.1)
    ax1.set_xlim(0,2)
    ax1.set_ylim(0,2)
    plt.show()
    filename = f"{path2db.split('/')[-1].split('.')[0]}_heatmap"
    #fig.savefig(os.path.join("research_data", path2db.split('/')[-1].split('.')[0], filename), dpi=150)

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
    argparser = argparse.ArgumentParser("Analyze results of main program. Make and save plots. Create heatmap or use clustering on data stored in the database.")
    argparser.add_argument("-db", "--database", help="Path to database file.")
    argparser.add_argument("-hm", "--heatmap", help="Use this flag if want to make a heatmap from the database data.", action="store_true", default=False)
    argparser.add_argument("-c", "--config", help="Print configuration used for the video.", action="store_true", default=False)
    args = argparser.parse_args()
    if args.database is not None:
        checkDir(args.database)
    if args.config:
        printConfig(args.database)
    if args.heatmap:
        coordinates2heatmap(args.database)

if __name__ == "__main__":
    main()