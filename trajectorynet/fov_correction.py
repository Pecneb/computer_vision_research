import numpy as np
import cv2
from yaml import load, dump
from functools import lru_cache
from matplotlib import pyplot as plt
from typing import Tuple, List, Optional
from copy import deepcopy
from tqdm import tqdm

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from utility.dataset import load_dataset
from utility.plots import plot_one_trajectory

from dataManagementClasses import TrackedObject

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
    def __init__(
        self,
        video_frame: np.ndarray,
        google_maps_image: np.ndarray,
        distance: int,
        transform_params: Optional[str] = None,
    ) -> None:
        self.video_frame = video_frame.copy()
        self.google_maps_image = google_maps_image.copy()
        # load transform params if they exist, if not initialize them, then save them
        if transform_params:
            try:
                self.load_transform_params(transform_params)
            except OSError:
                self.initialize_transform_matrix()
                self.save_transform_params(transform_params)
        else:
            self.initialize_transform_matrix()
        self.initialize_origo()
        self.initialize_meter_per_pixel(distance)

    @staticmethod
    def get_meter_per_pixel(img: np.ndarray, distance: float) -> float:
        """Get meter per pixel.

        Parameters
        ----------
        img : ndarray
            The image to get the rectangle from.
        google_maps_image : ndarray
            The image to get the rectangle from.
        distance : float
            Distance in meters.

        Returns
        -------
        float
            Meter per pixel.
        """
        return distance / img.shape[1]

    @staticmethod
    def get_points(img: np.ndarray) -> Tuple[int, int, int, int]:
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

    def initialize_meter_per_pixel(self, distance: float) -> None:
        """Initialize meter per pixel.

        Parameters
        ----------
        distance : float
            Distance in meters.
        """
        self.meter_per_pixel = self.get_meter_per_pixel(
            self.google_maps_image, distance
        )

    def initialize_transform_matrix(self) -> None:
        """Initialize the transform matrix from the image."""
        self.pts1 = self.get_points(self.video_frame)
        self.pts2 = self.get_points(self.google_maps_image)
        self.M = cv2.getPerspectiveTransform(
            np.float32(self.pts1), np.float32(self.pts2)
        )

    def save_transform_params(self, path: str) -> None:
        """Save the transform matrix to a file."""
        np.savez(path, M=self.M, pts1=self.pts1, pts2=self.pts2)

    def load_transform_params(self, path: str) -> None:
        """Load the transform matrix from a file."""
        data = np.load(path)
        self.M, self.pts1, self.pts2 = data["M"], data["pts1"], data["pts2"]

    def initialize_origo(self) -> None:
        """Get geometric center of the second images pts2."""
        self.origo = (
            (self.pts2[0][0] + self.pts2[1][0] + self.pts2[2][0] + self.pts2[3][0]) / 4,
            (self.pts2[0][1] + self.pts2[1][1] + self.pts2[2][1] + self.pts2[3][1]) / 4,
        )

    def get_transformed_image(self) -> np.ndarray:
        """Get transformed image."""
        return cv2.warpPerspective(
            self.video_frame,
            self.M,
            (self.video_frame.shape[1], self.video_frame.shape[0]),
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

    def get_transformed_origo(self) -> Tuple[float, float]:
        """Get x and y coordinates from origo."""
        return self.get_xy(self.origo[0][0], self.origo[0][1])

    def get_shifted_xy(self, u: float, v: float) -> Tuple[float, float]:
        """Get shifted x and y coordinates from u and v coordinates.

        Parameters
        ----------
        u : float
            X coordinate in the original image.
        v : float
            Y coordinate in the original image.

        Returns
        -------
        Tuple[float, float]
            X,Y coordinates in the transformed image.
        """
        x_0, y_0 = self.origo
        x, y = self.get_xy(u, v)
        return x - x_0, y - y_0

    def get_shifted_xy_in_meters(self, u: float, v: float) -> Tuple[float, float]:
        """Get shifted x and y coordinates from u and v coordinates.

        Parameters
        ----------
        u : float
            X coordinate in the original image.
        v : float
            Y coordinate in the original image.
        meter_per_pixel : float
            Meter per pixel.

        Returns
        -------
        Tuple[float, float]
            X,Y coordinates in the transformed image.
        """
        x, y = self.get_shifted_xy(u, v)
        return x * self.meter_per_pixel, y * self.meter_per_pixel


def transform_trajectory(
    obj: TrackedObject, transformer: FOVCorrectionOpencv, upscale: bool = False
) -> TrackedObject:
    """Transform trajectory with geometric transform method of opencv.

    Parameters
    ----------
    obj : TrackedObject
        Original tracked object.
    transformer : FOVCorrectionOpencv
        Transformer object.

    Returns
    -------
    TrackedObject
        Transformed tracked object.
    """
    obj_transformed = deepcopy(obj)
    for i, det in enumerate(obj_transformed.history):
        det.X, det.Y = (
            transformer.get_shifted_xy(
                int(det.X * transformer.video_frame.shape[0]),
                int(det.Y * transformer.video_frame.shape[0]),
            )
            if upscale
            else transformer.get_shifted_xy(det.X, det.Y)
        )
        obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
    return obj_transformed


def transform_trajectories(
    objs: List[TrackedObject], transformer: FOVCorrectionOpencv, upscale: bool = False
) -> List[TrackedObject]:
    """Transform trajectories with geometric transform method of opencv.

    Parameters
    ----------
    objs : List[TrackedObject]
        Tracked objects.
    transformer : FOVCorrectionOpencv
        Transformer object.

    Returns
    -------
    List[TrackedObject]
        List of transformed tracked objects.
    """
    objs_transformed = []
    for obj in objs:
        objs_transformed.append(transform_trajectory(obj, transformer, upscale))
    return objs_transformed


def fov_correction_opencv():
    """FOV correction using OpenCV.
    dst(x,y)=src(M11x+M12y+M13/M31x+M32y+M33,M21x+M22y+M23/M31x+M32y+M33)
    """
    with open("config/fov_correction.yaml", "r") as f:
        config = load(f, Loader=Loader)
    dataset_path = config["dataset"]["path"]
    video_path = config["dataset"]["video"]
    google_maps_image_path = config["dataset"]["google_maps_image"]
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    google_maps_image = cv2.imread(google_maps_image_path)
    transformer = FOVCorrectionOpencv(frame, google_maps_image)
    # transformer.initialize_points()
    # transformer.initialize_origo()
    # transformer.initialize_meter_per_pixel(config["fov_correction"]["distance"])
    dst = transformer.get_transformed_image()
    plt.subplot(121), plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), plt.title(
        "Input"
    )
    plt.subplot(122), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.title(
        "Output"
    )
    plt.show()
    dataset = load_dataset(dataset_path)
    fig, ax = plt.subplots(ncols=2, figsize=(7, 7))
    for obj in tqdm(dataset[550:600]):
        obj_transformed = deepcopy(obj)
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = int(det.X * frame.shape[0]), int((det.Y) * frame.shape[0])
            # det.X, det.Y = int(det.X), int(det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[0])
        for i, det in enumerate(obj_transformed.history):
            det.X, det.Y = transformer.get_shifted_xy_in_meters(det.X, det.Y)
            # print(det.X, det.Y)
            obj_transformed.history_X[i], obj_transformed.history_Y[i] = det.X, det.Y
        plot_one_trajectory(obj_transformed, ax[1])
    plt.show()


def main():
    fov_correction_opencv()


if __name__ == "__main__":
    main()
