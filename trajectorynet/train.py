import warnings
from clustering import (
    make_4D_feature_vectors,
    clustering_on_feature_vectors,
    kmeans_mse_clustering,
    calc_cluster_centers,
)
from fov_correction import transform_trajectories, FOVCorrectionOpencv
from classifier import OneVsRestClassifierWrapper
from utility.dataset import load_dataset
from utility.models import save_model, mask_labels
from utility.featurevector import FeatureVector
from utility.preprocessing import filter_trajectories
from utility.plots import plot_cross_validation_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import OPTICS
from sklearn.model_selection import cross_val_score, KFold, ParameterGrid
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from scipy.stats import stats
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
from logging import DEBUG, INFO, Logger, StreamHandler, getLogger
from typing import Union, Tuple, Dict, Any
from time import perf_counter
from datetime import datetime
import pandas as pd
import threading
import yaml
import cv2


cache_dir = os.path.join(os.path.dirname(__file__), "cache")


def warn(*args, **kwargs):
    pass


warnings.warn = warn

DEBUG_FLAG = os.getenv("DEBUG") == "1"
FPS = 30.0


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
    main_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset"
    )
    main_parser.add_argument("--enter-exit-threshold", type=float, default=0.4)
    main_parser.add_argument("--edge-distance-threshold", type=float, default=0.7)
    main_parser.add_argument("--detection-distance-threshold", type=float, default=0.01)
    main_parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        required=True,
        choices=["SVM", "KNN", "DT"],
        help="Model to train",
    )
    main_parser.add_argument(
        "--output", type=str, required=True, help="Path to save model"
    )
    main_parser.add_argument(
        "--feature-vector-version",
        nargs="+",
        choices=["1", "1_SY", "7", "7_SY"],
        default="1",
        help="Version of feature vector to use.",
    )
    main_parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of samples in a cluster.",
    )
    main_parser.add_argument(
        "--xi",
        type=float,
        default=0.05,
        help="Minimum difference between reachability distances.",
    )
    main_parser.add_argument(
        "--max-eps", type=float, default=0.15, help="Maximum reachability distance."
    )
    main_parser.add_argument(
        "--mse", type=float, default=0.2, help="Mean square error."
    )
    main_parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of jobs to run in parallel."
    )
    main_parser.add_argument(
        "--cross-validation",
        action="store_true",
        default=False,
        help="Run cross validation.",
    )
    main_parser.add_argument(
        "--grid-search",
        action="store_true",
        default=False,
        help="Run grid search.",
    )

    arguments = main_parser.parse_args()
    return arguments


def parse_config() -> Dict[str, Any]:
    with open("config/training_config.yaml", "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config


def make_classifier(
    model: str, n_jobs: int = -1, **estimator_args
) -> OneVsRestClassifierWrapper:
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
        classifier = SVC(probability=True, **estimator_args)
    elif model == "KNN":
        classifier = KNeighborsClassifier(**estimator_args)
    elif model == "DT":
        classifier = DecisionTreeClassifier(**estimator_args)
    elif model == "MLP":
        classifier = MLPClassifier(**estimator_args)
    else:
        raise ValueError(f"Unknown model: {model}")
    return OneVsRestClassifierWrapper(classifier, n_jobs=n_jobs, verbose=1)


def generate_feature_vectors(
    X: np.ndarray,
    Y: np.ndarray,
    Y_pooled: np.ndarray,
    version: str = "1",
    fps: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate feature vectors from X, Y and Y_pooled

    Parameters
    ----------
    X : np.ndarray
        Dataset
    Y : np.ndarray
        labels
    Y_pooled : np.ndarray
        pooled labels
    version : str, optional
        version number, by default "1"

    Returns
    -------
    np.ndarray

    """
    if version == "1":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.factory_1(
            trackedObjects=X, labels=Y, pooled_labels=Y_pooled, k=6
        )
    elif version == "1_SY":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.factory_1_SY(
            trackedObjects=X, labels=Y, pooled_labels=Y_pooled, k=6, scale_y=1.5
        )
    elif version == "1_SG_transform":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.factory_1_SG_fov_transform(
            trackedObjects=X, labels=Y, pooled_labels=Y_pooled, k=6, fps=fps
        )
    elif version == "7":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.factory_7(
            trackedObjects=X, labels=Y, pooled_labels=Y_pooled, max_stride=30
        )
    elif version == "7_SY":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.factory_7(
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
            weights=np.array([1, 1.5, 100, 150, 2, 3, 200, 300]),
        )
    elif version == "7_SG_transform":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.factory_7_SG_velocity(
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
            weights=np.array([1, 1.5, 100, 150, 2, 3, 200, 300]),
            fps=fps,
        )
    elif version == "Re":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.make_feature_vectors(
            FeatureVector.Re,
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
        )
    elif version == "ReVe":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.make_feature_vectors(
            FeatureVector.ReVe,
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
        )
    elif version == "ReVeAe":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.make_feature_vectors(
            FeatureVector.ReVeAe,
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
        )
    elif version == "ReRs":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.make_feature_vectors(
            FeatureVector.ReRs,
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
        )
    elif version == "ReVeRs":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.make_feature_vectors(
            FeatureVector.ReVeRs,
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
        )
    elif version == "ReVeAeRs":
        X_fv, Y_fv, Y_pooled_fv, _ = FeatureVector.make_feature_vectors(
            FeatureVector.ReVeAeRs,
            trackedObjects=X,
            labels=Y,
            pooled_labels=Y_pooled,
            max_stride=30,
        )
    return X_fv, Y_fv, Y_pooled_fv


def main():
    T0 = perf_counter()
    logger = init_logger()
    # args = get_arguments()
    args = parse_config()
    if args["debug"]:
        logger.setLevel(DEBUG)
    logger.debug(f"Arguments: {args}")
    # load dataset from path, either a single file or a directory
    dataset = load_dataset(args["dataset"])
    dataset = filter_trajectories(
        dataset,
        threshold=args["edge_distance_threshold"],
        enter_exit_dist=args["enter_exit_threshold"],
        detectionDistanceFiltering=False,
    )
    logger.debug(len(dataset))
    # extract enter and exit points of trajectories
    feature_vectors_clustering = make_4D_feature_vectors(dataset)
    logger.debug(len(feature_vectors_clustering))
    # cluster trajectories using OPTICS
    t0 = perf_counter()
    _, labels = clustering_on_feature_vectors(
        feature_vectors_clustering,
        OPTICS,
        min_samples=args["min_samples"],
        xi=args["xi"],
        max_eps=args["max_eps"],
    )
    logger.debug(f"Clustering took {perf_counter() - t0} seconds")
    # get class labels
    classes = np.unique(labels)
    logger.info(f"Classes: {classes}")
    # remove outliers
    X = dataset[labels != -1]
    Y = labels[labels != -1]
    logger.debug(f"X: {X.shape}, Y: {Y.shape}")
    # pool classes
    Y_pooled, _, _, _, pooled_classes = kmeans_mse_clustering(
        X, Y, n_jobs=args["n_jobs"], mse_threshold=args["mse"]
    )
    logger.debug(f"Y_pooled: {Y_pooled.shape}, pooled_classes: {pooled_classes}")
    logger.info(f"Pooled classes: {pooled_classes}")
    # calculate cluster centroids
    cluster_centers = calc_cluster_centers(X, Y)
    cluster_centers_pooled = calc_cluster_centers(X, Y_pooled)
    logger.debug(
        f"cluster_centers: {cluster_centers.shape}, cluster_centers_pooled: {cluster_centers_pooled.shape}"
    )
    FPS = 30.0
    if args["fov_correction"]:
        cap = cv2.VideoCapture(args["video_path"])
        ret, img = cap.read()
        img_google_maps = cv2.imread(args["google_map_image"])
        transform_params_path = os.path.join(args["output"], "transform_params.npz")
        transformer = FOVCorrectionOpencv(
            img, img_google_maps, args["distance"], transform_params_path
        )
        X = np.array(transform_trajectories(X, transformer, upscale=True))
        FPS = cap.get(cv2.CAP_PROP_FPS)
    # split dataset into train and test sets
    X_train, X_test, Y_train, Y_test, Y_pooled_train, Y_pooled_test = train_test_split(
        X, Y, Y_pooled, test_size=0.2, random_state=42
    )
    logger.debug(
        f"X_train: {X_train.shape}, Y_train: {Y_train.shape}, Y_pooled_train: {Y_pooled_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}, Y_pooled_test: {Y_pooled_test.shape}"
    )
    # # plot cross validation data
    # if args.cross_validation:
    #     logger.debug("Plotting cross validation data")
    #     fig, ax = plt.subplots()
    #     cv = KFold(n_splits=5)
    #     plot_cross_validation_data(cv, X_train, Y_train, ax, n_splits=5)
    #     fig.savefig(os.path.join(args.output, "cross_validation_data.png"))
    # for now generate only version 1 feature vectors
    # create dataframes for results
    cross_validation_results = pd.DataFrame(columns=["classifier", "version"])
    for i in range(5):
        cross_validation_results[f"split_{i} (percent)"] = []
    cross_validation_results["mean (percent)"] = []
    cross_validation_results["std (percent)"] = []
    results = pd.DataFrame(
        columns=[
            "classifier",
            "version",
            "balanced_test_score (percent)",
            "balanced_pooled_test_score (percent)",
            "time (s)",
        ]
    )
    if args["grid_search"]:
        param_grid = {
            "DT": {"max_depth": [None, 10, 15]},
            "KNN": {"n_neighbors": [3, 5, 10]},
            "SVM": {"C": [10, 100, 1000], "max_iter": [30000]},
            "MLP": {
                "hidden_layer_sizes": [(100, 100), (100, 100, 100, 100)],
                "solver": ["adam"],
                "learning_rate": ["adaptive"],
                "max_iter": [1500, 3000],
                "alpha": [0.01, 0.1],
                "learning_rate_init": [0.001, 0.01],
                "verbose": [False]
            },
        }
    else:
        param_grid = {
            "DT": {"max_depth": [None]},
            "KNN": {"n_neighbors": [7]},
            "SVM": {"C": [100], "max_iter": [60000]},
            "MLP": {"hidden_layer_sizes": [(100, 100, 100, 100)], "solver": ["adam"], "learning_rate": ["adaptive"], "max_iter": [3000], "alpha": [0.01], "learning_rate_init": [0.001], "verbose": [False]}
        }
    for v in args["feature_vector_version"]:
        logger.debug(f"Version: {v}")
        X_fv_train, Y_fv_train, Y_pooled_fv_train = generate_feature_vectors(
            X_train, Y_train, Y_pooled_train, version=v, fps=FPS
        )
        X_fv_test, Y_fv_test, Y_pooled_fv_test = generate_feature_vectors(
            X_test, Y_test, Y_pooled_test, version=v, fps=FPS
        )
        logger.debug(
            f"X_fv_test: {X_fv_test.shape}, Y_fv_test: {Y_fv_test.shape}, Y_pooled_fv_test: {Y_pooled_fv_test.shape}"
        )
        logger.debug(f"X_fv_train: {X_fv_train[:20]}")
        for m in args["model"]:
            for params in ParameterGrid(param_grid[m]):
                try:
                    # create classifier
                    estimator = make_classifier(m, n_jobs=args["n_jobs"], **params)
                    logger.debug(f"Estimator: {estimator}")
                    # run cross validation
                    if args["cross_validation"]:
                        logger.debug("Plotting cross validation data")
                        fig, ax = plt.subplots()
                        cv = KFold(n_splits=5)
                        plot_cross_validation_data(
                            cv, X_fv_train, Y_fv_train, ax, n_splits=5
                        )
                        fig.savefig(
                            os.path.join(args["output"], "cross_validation_data.png")
                        )
                        t0 = perf_counter()
                        scores = cross_val_score(
                            estimator,
                            X_fv_train,
                            Y_fv_train,
                            cv=cv,
                            scoring="balanced_accuracy",
                            n_jobs=args["n_jobs"],
                        )
                        t1 = perf_counter()
                        logger.info(
                            f"Cross validation scores: {scores}, mean: {scores.mean()}, deviation: {scores.std()}"
                        )
                        entry_num = len(cross_validation_results)
                        cross_validation_results.loc[entry_num, "classifier"] = "".join(
                            [m, " ", str(params)]
                        )
                        cross_validation_results.loc[entry_num, "version"] = v
                        for i, s in enumerate(scores):
                            cross_validation_results.loc[
                                entry_num, "split_" + str(i) + " (percent)"
                            ] = (s * 100)
                        cross_validation_results.loc[entry_num, "mean (percent)"] = (
                            scores.mean() * 100
                        )
                        cross_validation_results.loc[entry_num, "std (percent)"] = (
                            scores.std() * 100
                        )
                    # train classifier and evaluate on test set
                    t0 = perf_counter()
                    # estimator = make_classifier(m, n_jobs=args["n_jobs"], **params)
                    estimator.fit(X_fv_train, Y_fv_train)
                    t1 = perf_counter()
                    Y_predicted = estimator.predict(X_fv_test)
                    Y_score = estimator.predict_proba(X_fv_test)
                    score = balanced_accuracy_score(y_true=Y_fv_test, y_pred=Y_predicted)
                    logger.info(f"Balanced Test score: {score}")
                    Y_predicted_pooled = mask_labels(Y_predicted, pooled_classes)
                    pooled_score = balanced_accuracy_score(
                        y_true=Y_pooled_fv_test, y_pred=Y_predicted_pooled
                    )
                    logger.info(f"Balanced Pooled Test score: {pooled_score}")
                    # calculate confusion matrix
                    confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(
                        y_true=Y_fv_test, y_pred=Y_predicted
                    )
                    confusion_matrix_display.ax_.set_title("Confusion Matrix")
                    logger.debug(f"Confusion matrix: {confusion_matrix_display}")
                    confusion_matrix_display.figure_.savefig(
                        os.path.join(args["output"], f"confusion_matrix_{v}_{m}.png")
                    )
                    confusion_matrix_display_pooled = (
                        ConfusionMatrixDisplay.from_predictions(
                            y_true=Y_pooled_fv_test, y_pred=Y_predicted_pooled
                        )
                    )
                    confusion_matrix_display_pooled.ax_.set_title(
                        "Confusion Matrix (Pooled)"
                    )
                    logger.debug(
                        f"Confusion matrix (pooled): {confusion_matrix_display_pooled}"
                    )
                    confusion_matrix_display_pooled.figure_.savefig(
                        os.path.join(args["output"], f"confusion_matrix_pooled_{v}_{m}.png")
                    )
                    # save model with additional data about dataset
                    estimator.cluster_centroids = cluster_centers
                    estimator.pooled_cluster_centroids = cluster_centers_pooled
                    estimator.pooled_classes = pooled_classes
                    save_model(
                        savedir=args["output"],
                        classifier_type=m,
                        model=estimator,
                        version=v,
                    )
                    results.loc[len(results)] = [
                        "".join([m, " ", str(params)]),
                        v,
                        score * 100,
                        pooled_score * 100,
                        t1 - t0,
                    ]
                except ValueError as e:
                    logger.error(f"Error: {e}")
                    continue
    print(cross_validation_results.to_markdown())
    print(results.to_markdown())
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args["cross_validation"]:
        cross_validation_results.to_csv(
            os.path.join(args["output"], f"cross_validation_results_{date}.csv")
        )
    results.to_csv(os.path.join(args["output"], f"results_{date}.csv"))
    print(f"Total time: {perf_counter() - T0} seconds")
    for thread in threading.enumerate():
        print(thread.name)
    sys.exit()


if __name__ == "__main__":
    main()
