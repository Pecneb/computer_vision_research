import os
import joblib
from classifier import OneVSRestClassifierExtended

def save_model(savedir: str, classifier_type: str, model: OneVSRestClassifierExtended = None):
    """Save model to research_data dir.

    Args:
        path2db (str): Path to database file. 
        classifier_type (str): Classifier name. 
        model (Model): The model itself. 
    """
    if not os.path.isdir(os.path.join(savedir, "models")):
        os.mkdir(os.path.join(savedir, "models"))
    savepath = os.path.join(savedir, "models")
    filename = os.path.join(savepath, f"{classifier_type}.joblib")
    if model is not None:
        joblib.dump(model, filename)
    else:
        print("Error: model is None, model was not saved.")

def load_model(path2model: str) -> OneVSRestClassifierExtended:
    """Load classifier model.

    Args:
        path2model (str): Path to model. 

    Returns:
        BinaryClassifier: Trained binary classifier model. 
    """
    return joblib.load(path2model)