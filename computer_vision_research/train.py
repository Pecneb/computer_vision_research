### Standard imports ###
import sys
from logging import getLogger, INFO, DEBUG, StreamHandler, Logger

### Third party imports ###
import argparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import OPTICS, KMeans

### Local imports ###
from utility.dataset import load_dataset
from utility.models import save_model


def init_logger() -> Logger:
    _logger = getLogger(__name__)
    _logger.setLevel(DEBUG)
    _stream_handler = StreamHandler(sys.stdout)
    _logger.addHandler(_stream_handler)
    return _logger


def args() -> argparse.Namespace:
    main_parser = argparse.ArgumentParser(description="Train a model")
    main_parser.add_argument("--dataset", type=str,
                             required=True, help="Path to dataset")
    main_parser.add_argument("--model", type=str, required=True,
                             choices=["SVM", "KNN", "DT"], help="Model to train")
    main_parser.add_argument("--output", type=str,
                             required=True, help="Path to save model")
    main_parser.add_argument(
        "--min-samples", type=int, default=100, help="Minimum number of samples in a cluster.")
    main_parser.add_argument("--xi", type=float, default=0.05,
                             help="Minimum difference between reachability distances.")
    main_parser.add_argument(
        "--max-eps", type=float, default=0.15, help="Maximum reachability distance.")
    main_parser.add_argument(
        "--metric", type=str, default="minkowski", help="Distance metric.")
    main_parser.add_argument(
        "-mse", "--mean-square-error", type=float, default=0.2, help="Mean square error.")
    arguments = main_parser.parse_args()
    return arguments


def main():
    logger = init_logger()
    _args = args()
    logger.debug(f"Arguments: {_args}")
    dataset = load_dataset(_args.dataset)


if __name__ == "__main__":
    main()
