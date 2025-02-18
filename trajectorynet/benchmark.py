import sys
import os
import argparse
import random
from typing import Tuple, Dict, Any
from time import perf_counter, process_time
import yaml
import cv2
import numpy as np
import warnings
from logging import DEBUG, INFO, Logger, StreamHandler, getLogger
from pathlib import Path
from sklearn.cluster import OPTICS
from memory_profiler import profile
from utility.models import load_model
from utility.dataset import load_dataset
from classifier import OneVsRestClassifierWrapper
from fov_correction import transform_trajectories, FOVCorrectionOpencv
from train import generate_feature_vectors

import matplotlib.pyplot as plt

from utility.preprocessing import (
    enter_exit_distance_filter,
    edge_distance_filter,
    fill_trajectories,
)
from clustering import (
    make_4D_feature_vectors,
    clustering_on_feature_vectors,
    kmeans_mse_clustering,
    calc_cluster_centers,
)

DEBUG_ENV = os.getenv("DEBUG", False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/benchmark_config.yaml")
    return parser.parse_args()

def parse_config(config_path: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return True, config
    except Exception as e:
        print(f"Error loading config: {e}")
        return False, {}


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def init_logger(log_level: int) -> Logger:
    logger = getLogger(__name__)
    logger.setLevel(log_level)
    handler = StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger


@profile
def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    return model.predict_proba(X)


def main():
    # initialize logger
    logger = init_logger(DEBUG if DEBUG_ENV else INFO)

    # load config file
    config_path = get_args().config
    if not Path(config_path).exists():
        logger.error("Config file does not exist")
        sys.exit(1)
    config_suc, config = parse_config(config_path)
    if not config_suc:
        logger.error("Failed to load config")
        sys.exit(1)
    logger.info(f"Config: {config}")

    # load models
    model_paths = [
        Path(
            config["dataset"],
            "models",
            model_name + "_" + config["feature_vector_version"][0] + ".joblib",
        )
        for model_name in config["model"]
        if Path(
            config["dataset"],
            "models",
            model_name + "_" + config["feature_vector_version"][0] + ".joblib",
        ).exists()
    ]
    # models: Dict[str, OneVsRestClassifierWrapper] = {
    #     str(model_path): load_model(model_path)[1]
    #     for model_path in model_paths
    #     if load_model(model_path)[0]
    # }
    items = list(model_paths)
    random.shuffle(items)
    for model_path in items:
        logger.info(f"Model loaded from {model_path}")
        # logger.info(f"Model: {model}")

    # load dataset
    dataset = load_dataset(config["dataset"])
    logger.info(f"Dataset: {dataset.shape}")
    # filter dataset by enter exit distance
    if config["preprocessing"]["enter_exit_distance"]["switch"]:
        dataset = enter_exit_distance_filter(
            trackedObjects=dataset,
            threshold=config["preprocessing"]["enter_exit_distance"]["threshold"],
        )
    # filter dataset by image edge distance
    if config["preprocessing"]["edge_distance"]["switch"]:
        dataset = edge_distance_filter(
            trackedObjects=dataset,
            threshold=config["preprocessing"]["edge_distance"]["threshold"],
        )
    # fill/interpolate trajectories that have missing frames
    if config["preprocessing"]["filling"]:
        dataset = fill_trajectories(dataset)
    logger.info(f"Dataset after preprocessing: {dataset.shape}")

    # generate feature vectors for clustring
    feature_vectors_4D = make_4D_feature_vectors(dataset)

    # run OPTICS clustring on generated feature vectors
    _, labels = clustering_on_feature_vectors(
        feature_vectors_4D,
        estimator=OPTICS,
        min_samples=config["min_samples"],
        xi=config["xi"],
        max_eps=config["max_eps"],
    )
    logger.info(f"Labels: {labels.shape}")

    # throw away unclustered data
    X = dataset[labels != -1]
    Y = labels[labels != -1]
    logger.info(f"X: {X.shape}, Y: {Y.shape}")

    # run KMeans clustring on clusters to further reduce the number of classes
    Y_pooled, _, _, _, pooled_classes = kmeans_mse_clustering(
        X, Y, n_jobs=config["n_jobs"], mse_threshold=config["mse"]
    )
    # calculate the new cluster centers
    # cluster_centers = calc_cluster_centers(X, Y)
    # cluster_centers_pooled = calc_cluster_centers(X, Y_pooled)

    # run geometric transformation on trajectories so they are converted from pixel space to real world space
    FPS = 30.0
    if config["fov_correction"]:
        cap = cv2.VideoCapture(config["video_path"])
        ret, img = cap.read()
        img_google_maps = cv2.imread(config["google_map_image"])
        transform_params_path = os.path.join(config["output"], "transform_params.npz")
        transformer = FOVCorrectionOpencv(
            img, img_google_maps, config["distance"], transform_params_path
        )
        X = np.array(transform_trajectories(X, transformer, upscale=True))
        FPS = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Feature vector version: {config['feature_vector_version'][0]}")
    # generate feature vectors for inference
    X_fv_test, Y_fv_test, Y_pooled_fv_test = generate_feature_vectors(
        X,
        Y,
        Y_pooled,
        version=config["feature_vector_version"][0],
        fps=FPS,
        max_stride=config["max_stride"],
        window_length=config["window_length"],
    )
    logger.info(f"Number of feature vectors: {X_fv_test.shape}")

    # run inference on models
    # benchmark the time and memory usage
    for model_name in items:
        logger.info(f"Model: {model_name}")
        suc, model = load_model(model_name)
        if not suc:
            continue
        t_p0 = process_time()
        # model.predict_proba(X_fv_test)
        run_inference(model, X_fv_test)
        t_p1 = process_time()
        print(f"Time taken: {t_p1 - t_p0} seconds (process time)")


if __name__ == "__main__":
    main()
