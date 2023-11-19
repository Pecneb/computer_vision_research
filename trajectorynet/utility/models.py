import os
import numpy as np
from typing import Any, Optional

import joblib
from .logging import init_logger

_logger = init_logger(__name__)


def save_model(savedir: str, classifier_type: str, model, version: Optional[str] = None) -> bool:
    """Save ML Model.

    Parameters
    ----------
    savedir : str
        Path to directory where the model should be placed.
    classifier_type : str
        Type of classifier, eg. KNN, DT, SVM, etc...
    model : OneVSRestClassifierExtended
        The model object itself.
    version : Optional[str], optional
        Version string, eg. 1, 7, 8SG, etc..., by default None

    Returns
    -------
    bool
        Return True if saving was successful, False otherwise.
    """
    if not os.path.isdir(os.path.join(savedir, "models")):
        os.mkdir(os.path.join(savedir, "models"))
    savepath = os.path.join(savedir, "models")
    if version is not None:
        filename = os.path.join(
            savepath, f"{classifier_type}_{version}.joblib")
    else:
        filename = os.path.join(savepath, f"{classifier_type}.joblib")
    try:
        joblib.dump(model, filename)
    except Exception:
        return False
    return True


def load_model(path2model: str) -> Any:
    """
    Load model from disk.

    Parameters
    ----------
    path2model : str
        Path to model

    Returns
    -------
    Any
        Model object
    """
    return joblib.load(path2model)


def mask_labels(Y_1: np.ndarray, Y_mask: np.ndarray) -> np.ndarray:
    """Mask Y_1 labels using Y_mask

    Parameters
    ----------
    Y_1 : np.ndarray
        Array of labels.
    Y_mask : np.ndarray
        A mask of Y_1's classes. For example there are 11 classes in Y_1 but only 3 classes in Y_mask. So the 11 classes have to be mapped to 3 classes. The mapping is done by the index of the class in Y_mask. For example if the first class in Y_mask is 3 then all the 1s in Y_1 will be mapped to 3s.

    Returns
    -------
    np.ndarray
        The masked labels.
    """
    Y_masked = np.zeros_like(Y_1)
    for i, label in enumerate(Y_1):
        Y_masked[i] = Y_mask[label]
    return Y_masked


def mask_predictions(Y: np.ndarray, Y_mask: np.ndarray) -> np.ndarray:
    """Mask Y labels using Y_mask
    Y is a 2D array of shape (n_samples, n_classes)
    Each row is a probability distribution over the classes.

    Parameters
    ----------
    Y : np.ndarray
        Predicted probabilities.
    Y_mask : np.ndarray
        The corresponding pooled labels of the original classes.

    Returns
    -------
    np.ndarray
        Pooled predicted probabilities.
    """
    pooled_classes = np.unique(Y_mask)
    Y_pooled = np.zeros(shape=(Y.shape[0], len(pooled_classes)))
    for i, row in enumerate(Y):
        for j, proba in enumerate(row):
            Y_pooled[i, Y_mask[j]] = proba if Y_pooled[i, Y_mask[j]
                                                       ] < proba else Y_pooled[i, Y_mask[j]]
    return Y_pooled