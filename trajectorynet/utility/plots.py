import cv2
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import BaseCrossValidator

from . import databaseLoader


def savePlot(fig: plt.Figure, name: str):
    fig.savefig(name, dpi=150)


def cvCoord2npCoord(Y: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV coordinates to numpy coordinates.

    Parameters
    ----------
    Y : np.ndarray
        OpenCV coordinates

    Returns
    -------
    np.ndarray
        Numpy coordinates
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


def plot_misclassified(misclassifiedTracks: List, output: str = None):
    """
    Plot the misclassified tracks.

    Parameters
    ----------
    misclassifiedTracks : list
        A list of misclassified tracks.
    output : str, optional
        The output directory to save the plot.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_misclassified(misclassifiedTracks, output="output_dir")
    """

    X_enter = [t.history_X[0] for t in misclassifiedTracks]
    Y_enter = [t.history_Y[0] for t in misclassifiedTracks]
    X_exit = [t.history_X[-1] for t in misclassifiedTracks]
    Y_exit = [t.history_Y[-1] for t in misclassifiedTracks]
    X_traj = np.ravel([t.history_X[1:-1] for t in misclassifiedTracks])
    Y_traj = np.ravel([t.history_Y[1:-1] for t in misclassifiedTracks])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X_enter, Y_enter, s=10, c='g')
    ax.scatter(X_exit, Y_exit, s=10, c='r')
    ax.scatter(X_traj, Y_traj, s=5, c='b')
    fig.show()
    if output is not None:
        _output = Path(output) / "plots"
        _output.mkdir(exist_ok=True)
        fig.savefig(fname=(_output / "misclassified.png"))


def plot_misclassified_feature_vectors(misclassifiedFV: np.ndarray, output: str = None, background: str = None, classifier: str = "SVM"):
    """
    Plot the misclassified feature vectors.

    Parameters
    ----------
    misclassifiedFV : np.ndarray
        The misclassified feature vectors.
    output : str, optional
        The output directory to save the plot, by default None.
    background : str, optional
        The path to the background image, by default None.
    classifier : str, optional
        The name of the classifier, by default "SVM".

    Returns
    -------
    None
        The function only generates and displays the plot.

    """
    X_mask = [False, False, False, False,
              False, False, True, False, False, False]
    Y_mask = [False, False, False, False,
              False, False, False, True, False, False]
    X = np.ravel([f[X_mask] for f in misclassifiedFV])
    Y = np.ravel([f[Y_mask] for f in misclassifiedFV])
    fig, ax = plt.subplots(figsize=(7, 7))
    if background is not None:
        I = plt.imread(fname=background)
        ax.imshow(I, alpha=0.4, extent=[0, 1280, 0, 720])
    ax.scatter((X * I.shape[1]) / (I.shape[1] / I.shape[0]),
               (1-Y) * I.shape[0], s=0.05, c='r')
    ax.set_xlim(left=0, right=1280)
    ax.set_ylim(bottom=0, top=720)
    ax.grid(visible=True)
    ax.set_title(label=f"{classifier} misclassifications")
    fig.show()
    if output is not None:
        _output = Path(output) / "plots"
        _output.mkdir(exist_ok=True)
        fig.savefig(fname=(_output / f"{classifier}_misclassified.png"))


def plot_one_prediction(
        bbox: np.ndarray, cluster_center: np.ndarray,
        im: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), line_thickness: int = 2):
    """
    Draw one prediction.

    Parameters
    ----------
    bbox : np.ndarray
        Bounding box.
    cluster_center : np.ndarray
        Center coordinates of the predicted cluster.
    im : np.ndarray
        Image to draw on.

    Notes
    -----
    This function is used to draw one prediction on the image.

    Examples
    --------
    >>> draw_one_prediction(bbox, cluster_center, im)
    """
    x, y, w, h = bbox
    x_c, y_c = cluster_center
    cv2.line(img=im, pt1=(int(x+w/2), int(y+h/2)),
             pt2=(int(x_c), int(y_c)), color=color, thickness=line_thickness)


def plot_one_cluster(cluster_center: np.ndarray, im: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), radius: int = 3, line_thickness: int = 2):
    """
    Draw one cluster.

    Parameters
    ----------
    cluster_center : np.ndarray
        Center coordinates of the predicted cluster.
    color : Tuple[int, int, int], optional
        Color of the cluster, by default (0, 0, 255).
    line_thickness : int, optional
        Thickness of the line, by default 2.

    Notes
    -----
    This function is used to draw one cluster on the image.

    Examples
    --------
    >>> draw_one_cluster(cluster_center)
    """
    x_c, y_c = cluster_center
    cv2.circle(img=im, center=(int(x_c), int(y_c)),
               radius=radius, color=color, thickness=line_thickness)


def plot_cross_validation_data(cv: BaseCrossValidator, X: np.ndarray, y: np.ndarray, ax: plt.Axes, n_splits: int = 5, line_width: int = 10):
    """Visualize cross-validation data.
    Plot the indices of the training and test sets generated by cross-validation.

    Parameters
    ----------
    cv : BaseCrossValidator
        Cross validation object.
    X : np.ndarray
        Input data. 
    y : np.ndarray
        Labels.
    n_splits : int, optional
        Cross-validation splits, by default 5
    line_width : int, optional
        Witdth of plotted line, by default 10
    """
    cmap_data = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm

    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        ax.scatter(
            range(len(indices)), [ii + .5] * len(indices),
            c=indices, marker='_', lw=line_width, cmap=cmap_cv,
            vmin=-.2, vmax=1.2)

    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y[np.argsort(y)], marker='_', lw=line_width, cmap=cmap_data)

    yticklabels = list(range(n_splits)) + ['class']
    ax.set(
        yticks=np.arange(n_splits+1) + .5, yticklabels=yticklabels,
        xlabel='Sample index', ylabel="CV iteration",
        ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)

    return ax

def plot_one_trajectory(trajectory, ax, color="b", label="Trajectory"):
    """Plot one trajectory.

    Parameters
    ----------
    trajectory : TrackedObject
        Trajectory to plot.
    ax : plt.Axes
        Axes to plot on.
    color : str, optional
        Color of the trajectory, by default "b"
    label : str, optional
        Label of the trajectory, by default "Trajectory"
    """
    ax.plot(trajectory.history_X, trajectory.history_Y, color=color, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectory")
    # ax.legend(loc="lower right")