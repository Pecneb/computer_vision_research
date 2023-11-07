import os
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
