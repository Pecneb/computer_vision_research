import numpy as np
from yaml import load, dump
from functools import lru_cache
from matplotlib import pyplot as plt
from typing import Tuple
from copy import deepcopy

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from utility.dataset import load_dataset
from utility.plots import plot_one_trajectory

"""
x_s = x / width * (width / height)
x = x_s / (width / height) * width
x = x_s * (height / width) * width
"""


class FOVCorrection:
    def __init__(self, config):
        self.config = config
        self.alfa_zero = config["alfa_zero"]
        self.delta = config["delta"]
        self.distance = config["distance"]
        self.p_u = config["resolution"]["horizontal"]
        self.p_v = config["resolution"]["vertical"]
        self.h = ((self.distance / 2) / np.sin(self.delta / 2)) * np.sin(self.alfa_zero)
        self.s_p = self.delta / self.p_u
        self.d_p_sin_alfa = (self.distance / self.p_u) * np.sin(self.alfa_zero)
        self.alfa = self.alfa_zero - (np.arange(self.p_v) * self.s_p)
        self.alfa = np.where(self.alfa < 10, 10, self.alfa)
        self.sin_alfa = np.sin(self.alfa)
        self.tan_alfa = np.tan(self.alfa)
        self.tan_alfa_zero = np.tan(self.alfa_zero)

    def __repr__(self) -> str:
        return f"FOVCorrection(alfa_zero={self.alfa_zero}, delta={self.delta}, \
            distance={self.distance}, p_u={self.p_u}, h={self.h}, s_p={self.s_p}), \
            d_p_sin_alfa={self.d_p_sin_alfa}, alfa={self.alfa}, sin_alfa={self.sin_alfa}, \
            tan_alfa={self.tan_alfa}, tan_alfa_zero={self.tan_alfa_zero})"

    def get_coord(self, u, v) -> Tuple[float, float]:
        x = u * self.d_p_sin_alfa / self.sin_alfa[v]
        y = self.h - ((1 / self.tan_alfa[v]) * (1 / self.tan_alfa_zero))
        return x, y


def main():
    with open("config/fov_correction.yaml", "r") as f:
        config = load(f, Loader=Loader)
    dataset_path = config["dataset"]["path"]
    corrector = FOVCorrection(config["fov_correction"])
    print(corrector)
    dataset = load_dataset(dataset_path)
    fig, ax = plt.subplots(ncols=2, figsize=(7, 7))
    dataset_transformed = []
    for obj in dataset[:10]:
        obj_transformed = deepcopy(obj)
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = int(det.X * corrector.p_v), int(det.Y * corrector.p_v)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[0])
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = corrector.get_coord(det.X, det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[1])
    plt.show()


if __name__ == "__main__":
    main()
