import numpy as np
import cv2
from yaml import load, dump
from functools import lru_cache
from matplotlib import pyplot as plt
from typing import Tuple
from copy import deepcopy
from tqdm import tqdm

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


class FOVCorrectionAndras:
    def __init__(self, config):
        self.config = config
        self.alfa_zero = np.radians(config["alfa_zero"])
        self.delta = np.radians(config["delta"])
        self.distance = config["distance"]
        self.p_u = config["resolution"]["horizontal"]
        self.p_v = config["resolution"]["vertical"]
        self.h = ((self.distance / 2) / np.sin(self.delta / 2)) * np.sin(self.alfa_zero)
        self.s_p = self.delta / self.p_u
        self.d_p_sin_alfa = (self.distance / self.p_u) * np.sin(self.alfa_zero)
        self.alfa = self.alfa_zero - (np.arange(self.p_v) * self.s_p)
        self.alfa = np.where(self.alfa < 0.1, 0.1, self.alfa)
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
        y = self.h * ((1 / self.tan_alfa[v]) - (1 / self.tan_alfa_zero))
        return x, y


def fov_correction_andras():
    with open("config/fov_correction.yaml", "r") as f:
        config = load(f, Loader=Loader)
    dataset_path = config["dataset"]["path"]
    corrector = FOVCorrectionAndras(config["fov_correction"])
    print(corrector)
    dataset = load_dataset(dataset_path)
    fig, ax = plt.subplots(ncols=2, figsize=(7, 7))
    dataset_transformed = []
    for obj in tqdm(dataset[500:700]):
        obj_transformed = deepcopy(obj)
        for i, det in enumerate(obj_transformed.history):
            # det.X, det.Y = int(det.X * corrector.p_v), int(det.Y * corrector.p_v)
            det.X, det.Y = int(det.X), int(det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[0])
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = corrector.get_coord(det.X, det.Y)
            # print(det.X, det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[1])
    plt.show()


class FOVCorrectionOpencv:
    def __init__(self, img: np.ndarray) -> None:
        self.img = img.copy()

    @staticmethod
    def get_rectangle(img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get rectangle coordinates from the image.

        Parameters
        ----------
        img : ndarray
            The image to get the rectangle from.
        """
        rectangle_points = []
        img_clone = img.copy()

        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                rectangle_points.append((x, y))
                cv2.circle(img_clone, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("image", img_clone)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_rectangle)

        while True:
            cv2.imshow("image", img_clone)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()
        return rectangle_points

    def initialize_points(self) -> None:
        """Initialize points from the image."""
        self.pts1 = self.get_rectangle(self.img)
        self.pts2 = self.get_rectangle(self.img)
        self.M = cv2.getPerspectiveTransform(
            np.float32(self.pts1), np.float32(self.pts2)
        )
    
    def get_transformed_image(self) -> np.ndarray:
        """Get transformed image."""
        return cv2.warpPerspective(
            self.img, self.M, (self.img.shape[1], self.img.shape[0])
        )

    def get_xy(self, u, v) -> Tuple[float, float]:
        """Get x and y coordinates from u and v coordinates.
        dst(x,y)=src(M11x+M12y+M13/M31x+M32y+M33,M21x+M22y+M23/M31x+M32y+M33)

        Parameters
        ----------
        u : float
            u coordinate.
        v : float
            v coordinate.
        """
        return (
            (self.M[0][0] * u + self.M[0][1] * v + self.M[0][2])
            / (self.M[2][0] * u + self.M[2][1] * v + self.M[2][2]),
            (self.M[1][0] * u + self.M[1][1] * v + self.M[1][2])
            / (self.M[2][0] * u + self.M[2][1] * v + self.M[2][2]),
        )


def fov_correction_opencv():
    """FOV correction using OpenCV.
    dst(x,y)=src(M11x+M12y+M13/M31x+M32y+M33,M21x+M22y+M23/M31x+M32y+M33)
    """
    with open("config/fov_correction.yaml", "r") as f:
        config = load(f, Loader=Loader)
    dataset_path = config["dataset"]["path"]
    video_path = config["dataset"]["video"]
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    print(frame.shape)
    transformer = FOVCorrectionOpencv(frame)
    transformer.initialize_points() 
    dst = transformer.get_transformed_image()
    print(transformer.pts1, transformer.pts2)
    print(transformer.M)
    # plt.subplot(121), plt.imshow(frame), plt.title("Input")
    # plt.subplot(122), plt.imshow(dst), plt.title("Output")
    # plt.show()
    dataset = load_dataset(dataset_path)
    fig, ax = plt.subplots(ncols=2, figsize=(7, 7))
    for obj in tqdm(dataset[500:600]):
        obj_transformed = deepcopy(obj)
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = int(det.X * frame.shape[0]), int(det.Y * frame.shape[0])
            # det.X, det.Y = int(det.X), int(det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[0])
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = transformer.get_xy(det.X, det.Y)
            # print(det.X, det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[1])
    plt.show()



def main():
    fov_correction_opencv()


if __name__ == "__main__":
    main()
