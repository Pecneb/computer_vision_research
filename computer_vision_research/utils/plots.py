import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

import computer_vision_research.utils.databaseLoader as databaseLoader
from computer_vision_research.dataManagementClasses import (
    TrackedObject
)

def savePlot(fig: plt.Figure, name: str):
    fig.savefig(name, dpi=150)

def cvCoord2npCoord(Y: np.ndarray) -> np.ndarray:
    """Convert OpenCV Y axis coordinates to numpy coordinates.

    Args:
        Y (np.ndarray): Y axis coordinate vector

    Returns:
        np.ndarray: Y axis coordinate vector
    """
    return 1 - Y

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

def plot_misclassified(misclassifiedTracks: List[TrackedObject], output: str = None):
    """Plot misclassified trajectories. If output is given, save plot to output/plots/misclassified.png.

    Args:
        misclassifiedTracks (List[dataManagementClasses.TrackedObject]): List of TrackedObjects, which was misclassified by an estimator.
        output (str, optional): Output directory path. Defaults to None.
    """
    X_enter = [t.history_X[0] for t in misclassifiedTracks]
    Y_enter = [t.history_Y[0] for t in misclassifiedTracks]
    X_exit = [t.history_X[-1] for t in misclassifiedTracks]
    Y_exit = [t.history_Y[-1] for t in misclassifiedTracks]
    X_traj = np.ravel([t.history_X[1:-1] for t in misclassifiedTracks])
    Y_traj = np.ravel([t.history_Y[1:-1] for t in misclassifiedTracks])
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(X_enter, Y_enter, s=10, c='g')
    ax.scatter(X_exit, Y_exit, s=10, c='r')
    ax.scatter(X_traj, Y_traj, s=5, c='b')
    fig.show()
    if output is not None:
        _output = Path(output) / "plots"
        _output.mkdir(exist_ok=True)
        fig.savefig(fname=(_output / "misclassified.png"))

def plot_misclassified_feature_vectors(misclassifiedFV: np.ndarray, output: str = None, background: str = None, classifier: str = "SVM"):
    """Plot misclassified trajectories. If output is given, save plot to output/plots/misclassified.png.

    Args:
        misclassifiedTracks (List[dataManagementClasses.TrackedObject]): List of TrackedObjects, which was misclassified by an estimator.
        output (str, optional): Output directory path. Defaults to None.
        background (str, optional): Background image of plot for better visualization.
    """
    X_mask = [False, False, False, False, False, False, True, False, False, False]
    Y_mask = [False, False, False, False, False, False, False, True, False, False]
    X = np.ravel([f[X_mask] for f in misclassifiedFV])
    Y = np.ravel([f[Y_mask] for f in misclassifiedFV])
    fig, ax = plt.subplots(figsize=(7,7))
    if background is not None:
        print(background)
        I = plt.imread(fname=background)
        # I = plt.imread("/media/pecneb/4d646cbd-cce0-42c4-bdf5-b43cc196e4a1/gitclones/computer_vision_research/research_data/Bellevue_150th_Newport_24h_v2/Preprocessed/Bellevue_150th_Newport.JPG", format="jpg")
        ax.imshow(I, alpha=0.4, extent=[0, 1280, 0, 720])
    ax.scatter((X * I.shape[1]) / (I.shape[1] / I.shape[0]), (1-Y) * I.shape[0], s=0.05, c='r')
    ax.set_xlim(left=0, right=1280)
    ax.set_ylim(bottom=0, top=720)
    ax.grid(visible=True)
    ax.set_title(label=f"{classifier} misclassifications")
    fig.show()
    if output is not None:
        _output = Path(output) / "plots"
        _output.mkdir(exist_ok=True)
        fig.savefig(fname=(_output / f"{classifier}_misclassified.png"))