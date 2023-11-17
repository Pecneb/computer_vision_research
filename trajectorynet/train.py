import warnings
from clustering import make_4D_feature_vectors, clustering_on_feature_vectors, kmeans_mse_clustering, calc_cluster_centers
from classifier import OneVsRestClassifierWrapper
from utility.dataset import load_dataset
from utility.models import save_model
from utility.featurevector import FeatureVector
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import OPTICS
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import argparse
import sys
import os
from logging import DEBUG, INFO, Logger, StreamHandler, getLogger
from typing import Union


def warn(*args, **kwargs):
    pass


warnings.warn = warn

DEBUG_FLAG = os.getenv("DEBUG") == "1"


def init_logger() -> Logger:
    """Initialize logger

    Returns
    -------
    Logger
        Logger object
    """
    _logger = getLogger(__name__)
    _logger.setLevel(DEBUG if DEBUG_FLAG else INFO)
    _stream_handler = StreamHandler(sys.stdout)
    _logger.addHandler(_stream_handler)
    return _logger


def get_arguments() -> argparse.Namespace:
    main_parser = argparse.ArgumentParser(description="Train a model")
    main_parser.add_argument("--dataset", type=str,
                             required=True, help="Path to dataset")
    main_parser.add_argument("--model", type=str, required=True,
                             choices=["SVM", "KNN", "DT"], help="Model to train")
    main_parser.add_argument("--output", type=str,
                             required=True, help="Path to save model")
    main_parser.add_argument("--feature-vector-version", choices=["1", "7"], default="1",
                             help="Version of feature vector to use.")
    main_parser.add_argument(
        "--min-samples", type=int, default=100, help="Minimum number of samples in a cluster.")
    main_parser.add_argument("--xi", type=float, default=0.05,
                             help="Minimum difference between reachability distances.")
    main_parser.add_argument(
        "--max-eps", type=float, default=0.15, help="Maximum reachability distance.")
    main_parser.add_argument(
        "--mse", type=float, default=0.2, help="Mean square error.")
    main_parser.add_argument("--n-jobs", type=int, default=-1,
                             help="Number of jobs to run in parallel.")
    arguments = main_parser.parse_args()
    return arguments


def make_classifier(model: str, n_jobs: int = -1, **estimator_args) -> OneVsRestClassifierWrapper:
    """Create OneVsRestClassifier with specified model

    Parameters
    ----------
    model : str
        Name of model to use

    Returns
    -------
    OneVsRestClassifier
        A OneVsRestClassifier with the specified model

    Raises
    ------
    ValueError
        If an unknown model is given as input raise ValueError
    """
    if model == "SVM":
        classifier = SVC(kernel="rbf", gamma="scale", probability=True)
    elif model == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model == "DT":
        classifier = DecisionTreeClassifier()
    else:
        raise ValueError(f"Unknown model: {model}")
    return OneVsRestClassifierWrapper(classifier, n_jobs=n_jobs, verbose=1)


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


def main():
    logger = init_logger()
    args = get_arguments()
    logger.debug(f"Arguments: {args}")
    # load dataset from path, either a single file or a directory
    dataset = load_dataset(args.dataset)
    logger.debug(len(dataset))
    # extract enter and exit points of trajectories
    feature_vectors_clustering = make_4D_feature_vectors(dataset)
    logger.debug(len(feature_vectors_clustering))
    # cluster trajectories using OPTICS
    _, labels = clustering_on_feature_vectors(feature_vectors_clustering, OPTICS, min_samples=args.min_samples,
                                              xi=args.xi, max_eps=args.max_eps)
    # get class labels
    classes = np.unique(labels)
    logger.info(f"Classes: {classes}")
    # remove outliers
    X = dataset[labels != -1]
    Y = labels[labels != -1]
    logger.debug(f"X: {X.shape}, Y: {Y.shape}")
    # pool classes
    Y_pooled, _, _, _, pooled_classes = kmeans_mse_clustering(
        X, Y, n_jobs=args.n_jobs, mse_threshold=args.mse)
    logger.debug(
        f"Y_pooled: {Y_pooled.shape}, pooled_classes: {pooled_classes}")
    # calculate cluster centroids
    cluster_centers = calc_cluster_centers(X, Y)
    cluster_centers_pooled = calc_cluster_centers(X, Y_pooled)
    logger.debug(
        f"cluster_centers: {cluster_centers.shape}, cluster_centers_pooled: {cluster_centers_pooled.shape}")
    # split dataset into train and test sets
    X_train, X_test, Y_train, Y_test, Y_pooled_train, Y_pooled_test = train_test_split(
        X, Y, Y_pooled, test_size=0.2, random_state=42)
    logger.debug(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, Y_pooled_train: {Y_pooled_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}, Y_pooled_test: {Y_pooled_test.shape}")
    # for now generate only version 1 feature vectors
    X_fv_train, Y_fv_train, Y_pooled_fv_train, _ = FeatureVector.factory_1(
        X_train, Y_train, Y_pooled_train, k=6)
    logger.debug(
        f"X_fv_train: {X_fv_train.shape}, Y_fv_train: {Y_fv_train.shape}, Y_pooled_fv_train: {Y_pooled_fv_train.shape}")
    X_fv_test, Y_fv_test, Y_pooled_fv_test, _ = FeatureVector.factory_1(
        X_test, Y_test, Y_pooled_test, k=6)
    logger.debug(
        f"X_fv_test: {X_fv_test.shape}, Y_fv_test: {Y_fv_test.shape}, Y_pooled_fv_test: {Y_pooled_fv_test.shape}")
    # create classifier
    estimator = make_classifier(args.model)
    logger.debug(f"Estimator: {estimator}")
    # run cross validation
    scores = cross_val_score(estimator, X_fv_train,
                             Y_fv_train, cv=5, scoring="balanced_accuracy")
    logger.info(
        f"Cross validation scores: {scores}, mean: {scores.mean()}, deviation: {scores.std()}")
    # train classifier and evaluate on test set
    estimator = make_classifier(args.model)
    estimator.fit(X_fv_train, Y_fv_train)
    Y_predicted = estimator.predict(X_fv_test)
    score = balanced_accuracy_score(y_true=Y_fv_test, y_pred=Y_predicted)
    logger.info(f"Balanced Test score: {score}")
    Y_predicted_pooled = mask_labels(Y_predicted, pooled_classes)
    pooled_score = balanced_accuracy_score(
        y_true=Y_pooled_fv_test, y_pred=Y_predicted_pooled)
    logger.info(f"Balanced Pooled Test score: {pooled_score}")
    estimator.cluster_centroids = cluster_centers
    estimator.pooled_cluster_centroids = cluster_centers_pooled
    estimator.pooled_classes = pooled_classes
    save_model(savedir=args.output, classifier_type=args.model, model=estimator, version=args.feature_vector_version)


if __name__ == "__main__":
    main()
