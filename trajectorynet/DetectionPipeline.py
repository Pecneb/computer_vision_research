"""
    Predicting trajectories of objects
    Copyright (C) 2022  Bence Peter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact email: ecneb2000@gmail.com
"""
from logging import DEBUG, INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal
from functools import lru_cache

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from classifier import OneVsRestClassifierWrapper
from dataManagementClasses import Detection as DarknetDetection
from dataManagementClasses import TrackedObject
from deep_sort.deep_sort.detection import Detection as DeepSORTDetection
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.tracker import Tracker
from joblib import dump
from masker import masker
from torch import nn
from utility.databaseLogger import logObject
from utility.models import load_model, mask_predictions
from utility.plots import plot_one_cluster, plot_one_prediction
from yolov7.models.common import Conv
from yolov7.models.experimental import Ensemble
from yolov7.utils.datasets import LoadImages, letterbox
from yolov7.utils.general import (check_img_size, check_imshow,
                                  non_max_suppression, scale_coords, xyxy2xywh)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized


class Yolov7(object):
    """Class to support Yolov7 Object Detection Pipeline.
    This class gives support for pre- and postprocessing images.
    Gives a high level interface to run inference with the yolov7 cnn pytorch model.

    Parameters
    ----------
    weights : str
        Path to weights file
    conf_thres : float, optional
        Confidence threshold, by default 0.6
    iou_thres : float, optional
        Intersection over Union threshold, by default 0.4
    imgsz : int, optional
        Input image size, by default 640
    stride : int, optional
        Convolution stride, by default 32
    half : bool, optional
        Half precision (from float32 to float16), by default True
    device : Union[int, str], optional
        Device id, by default 0
    batch_size : int, optional
        Size of input batch (this means that the model will run inference on batch_size images at once), by default 1
    debug : bool, optional
        Debug flag, by default True

    Attributes
    ----------
    _logger : Logger
        Yolo class logger object
    device : Device
        Pytorch device object, initialized from device id given in contructor args
    model : Model
        Loaded pytorch model
    stride : int
        Convolution stride, and resize stride
    imgsz : int
        Input image size
    conf_thres : float, optional
        Confidence threshold
    iou_thres : float, optional
        Intersection over Union threshold
    names : List[str]
        List of class/label names
    colors : List[Tuple(int, int, int)]
        Random colors for each class/label
    half : bool
        If half True half precision (float16) is used, this means faster inference but lower precision

    Methods
    -------
    _load(weights: str, imgsz: int = 640, batch_size: int = 1, device: str = "cuda") -> nn.Module
        Load weights into memory.
    preprocess(img0: np.ndarray) -> np.ndarray
        Preprocess raw image for inference.
    postprocess(pred: torch.Tensor, im0: np.ndarray, im: np.ndarray, show: bool = True) -> List
        Run NMS on infer output, then rescale them and create a vector with [label, conf, bbox]
    warmup()
        Warm up model.
    infer(img: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]
        Run inference on input images.

    """

    def __init__(self, weights: str, conf_thres: float = 0.6, iou_thres: float = 0.4, imgsz: int = 640, stride: int = 32, augment: bool = False, half: bool = True, device: Union[int, str] = "0", batch_size: int = 1, debug: bool = True) -> None:
        # region init logger
        self._logger = getLogger("Yolo_Logger")
        _logHandler = StreamHandler()
        if debug:
            self._logger.setLevel(DEBUG)
            _logHandler.setLevel(DEBUG)
        else:
            self._logger.setLevel(INFO)
            _logHandler.setLevel(INFO)
        _formatter = Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        _logHandler.setFormatter(_formatter)
        self._logger.addHandler(_logHandler)
        # endregion

        # region init yolo
        self.device = select_device(device, batch_size)
        self._logger.debug(f"Device: {self.device}")
        self.model = self._load(weights, imgsz, device)
        self._logger.debug(f"Model: {self.model}")
        self.stride = int(self.model.stride.max())  # model stride
        self._logger.debug(f"Stride: {self.stride}")
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self._logger.debug(f"Image size: {self.imgsz}")
        self.conf_thres = conf_thres
        self._logger.debug(f"Confidence threshold: {self.conf_thres}")
        self.iou_thres = iou_thres
        self._logger.debug(f"IoU threshold: {self.iou_thres}")
        self.names = self.model.module.names if hasattr(
            self.model, "module") else self.model.names
        self._logger.debug(f"Names: {self.names}")
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for _ in range(len(self.names))]
        self._logger.debug(f"Colors: {self.colors}")
        self.augment = augment
        self._logger.debug(f"Augment: {self.augment}")
        self.half = half
        if self.half:
            self.model.half()
        self._logger.debug(f"Half precision: {self.half}")
        # endregion

    def _load(self, weights: str, imgsz: int = 640, batch_size: int = 1, device: str = "cuda") -> nn.Module:
        """Load weights into memory.

        Parameters
        ----------
        weights : str
            Path to weights file.
        imgsz : int, optional
            Input image size, by default 640
        batch_size : int, optional
            Batch size, by default 1
        device : str, optional
            Device id, by default "cuda"

        Returns
        -------
        nn.Module
            Pytorch neural network module object
        """
        ckpt = torch.load(weights, map_location=device)
        model = Ensemble().append(ckpt['ema' if ckpt.get(
            'ema') else 'model'].float().fuse().eval())  # FP32 model
        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        return model[-1]

    def preprocess(self, img0: np.ndarray) -> np.ndarray:
        """Preprocess raw image for inference.
        Convert to NN input img size, into shape (3,640,640).
        Check for half precision flag (fp32 or fp16).
        Normalize img by dividing image pixel values with 255.
        If image's shape is (3,640,640) then add additional dim.

        Parameters
        ----------
        img0 : ndarray 
            Input image.

        Returns
        -------
        ndarray
            Preprocessed image of shape (3,640,640)
        """
        # Padded resize
        if len(img0.shape) == 3:  # check for single image
            self._logger.debug(f"Input image shape: {img0.shape}")
            img = letterbox(img0, new_shape=self.imgsz, stride=self.stride)[0]
            img = np.expand_dims(img, axis=0)
        else:  # check for multiple images as input
            img = [letterbox(x, new_shape=self.imgsz, stride=self.stride)[
                0] for x in img0]
        img = np.stack(img, 0)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # BGR to RGB, to 3x416x416
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        # store img in memory in on place, not in segments for faster lookup
        img = np.ascontiguousarray(img)
        """
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        """
        return img

    def postprocess(self, pred: torch.Tensor, im0: np.ndarray, im: np.ndarray, show: bool = True) -> List:
        """Run NMS on infer output, then rescale them and create a vector with [label, conf, bbox]

        Parameters
        ----------
        pred : pt.Tensor
            Predictions given by infer
        img : np.ndarray
            Preprocessed image

        Returns
        -------
        List
            List of shape (n,6), where n is the number of detections per image.
        """
        _pred = non_max_suppression(
            pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        detections_adjusted = []
        for det in _pred:
            if len(det):
                det[:, :4] = scale_coords(
                    im.shape[1:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    if show:
                        plot_one_box(xyxy, im0, label=self.names[int(
                            cls)], color=self.colors[int(cls)], line_thickness=3)
                    # convert to center coords, width, height format
                    bbox = xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                    label = self.names[int(cls)]
                    detections_adjusted.append(
                        [label, conf.item(), bbox[0, :].numpy()])
        return detections_adjusted

    def warmup(self):
        """Warm up model.
        """
        for _ in range(3):
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(
                self.device).type_as(next(self.model.parameters())))

    def infer(self, img: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Run inference on input images.

        Parameters
        ----------
        img : ndarray
            Input image

        Returns
        -------
        Tuple[Tensor, ndarray]
            Model output of shape and input image
        """
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():  # calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        return pred


class DeepSORT(object):
    """Detector class that can generate dataset for trajectory prediction model training.

    Parameters
    ----------
    source : Union[int, str], optional
        Video source, can be webcam number, video path, directory path containing videos, by default 0
    output : Optional[str], optional
        Output directory, by default None
    max_cosine_distance : float, optional
        Gating threshold for cosine distance metric (object appearance), by default 10.0
    max_iou_distance : float, optional
        Max intersection over union distance, by default 0.7
    nn_budget : float, optional
        Maximum size of the appearence descriptor gallery, by default 100
    historyDepth : int, optional
        Length of history, by default 30
    debug : bool, optional
        Debug flag, by default True

    Attributes
    ----------
    source : Union[int, str]
        Video source, can be webcam number, video path, directory path containing videos
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance)
    max_iou_distance : float
        Max intersection over union distance       
    nn_budget : float
        Maximum size of the appearence descriptor gallery
    historyDepth : int
        Length of history 

    Methods
    -------
    init_tracker_metric(max_cosine_distance: float, nn_budget: float, metric: str = "cosine") -> NearestNeighborDistanceMetric 
        Deepsort metric factory.
    tracker_factory(metric: NearestNeighborDistanceMetric, max_iou_distance: float, historyDepth: int) -> Tracker
        Create tracker object.
    make_detection_object(darknetDetection: DarknetDetection) -> DeepSORTDetection
        Wrap a DarknetDetection object into a DeepSORT Detection object.
    update_history(history: List[TrackedObject], new_detections: List[DarknetDetection], joblibbuffer: Optional[List[TrackedObject]] = None, db_connection: Optional[str] = None)
        Update trajectory history with new detections.

    """

    def __init__(self, max_cosine_distance: float = 10.0, max_iou_distance: float = 0.7, nn_budget: float = 100, historyDepth: int = 30, debug: bool = False) -> None:
        ### Logging ###
        self._logger = getLogger("DeepSORT_Logger")
        _logHandler = StreamHandler()
        if debug:
            self._logger.setLevel(DEBUG)
            _logHandler.setLevel(DEBUG)
        else:
            self._logger.setLevel(INFO)
            _logHandler.setLevel(INFO)
        _formatter = Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        _logHandler.setFormatter(_formatter)

        self.max_cosine_distance = max_cosine_distance
        self._logger.debug(f"Max cosine distance: {self.max_cosine_distance}")
        self.max_iou_distance = max_iou_distance
        self._logger.debug(f"Max IoU distance: {self.max_iou_distance}")
        self.nn_budget = nn_budget
        self._logger.debug(f"NN budget: {self.nn_budget}")
        self.historyDepth = historyDepth
        self._logger.debug(f"History depth: {self.historyDepth}")

        self._Metric = self.init_tracker_metric(
            max_cosine_distance=max_cosine_distance, nn_budget=nn_budget)
        self._logger.debug(f"Metric: {self._Metric}")
        self._Tracker = self.tracker_factory(
            metric=self._Metric, max_iou_distance=max_iou_distance, historyDepth=historyDepth)
        self._logger.debug(f"Tracker: {self._Tracker}")

    @staticmethod
    def init_tracker_metric(max_cosine_distance: float, nn_budget: float, metric: str = "cosine") -> NearestNeighborDistanceMetric:
        """Deepsort metric factory.

        Parameters
        ----------
        max_cosince_distance : float
            Garing threshold for cosine distance metric
        nn_budget : float
            Maximum size of the appearence descriptor gallery,
        metric : str, optional
            Metric type, by default "cosine"
        """
        return NearestNeighborDistanceMetric(metric, max_cosine_distance, nn_budget)

    @staticmethod
    def tracker_factory(metric: NearestNeighborDistanceMetric, max_iou_distance: float, historyDepth: int) -> Tracker:
        """Create tracker object. 

        Parameters
        ----------
        metric : NearestNeighborDistanceMetric
            Distance metric object
        max_iou_distance : float
        historyDepth : int
            Length of history.

        Returns
        -------
        Tracker
            Tracker object.
        """
        return Tracker(metric=metric, max_age=10, historyDepth=historyDepth, max_iou_distance=max_iou_distance)

    @staticmethod
    def make_detection_object(darknetDetection: DarknetDetection) -> DeepSORTDetection:
        """Wrap a DarknetDetection object into a DeepSORT Detection object.

        Parameters
        ----------
        darknetDetection : DarknetDetection
            Detection representing a darknet/yolo detection.

        Returns
        -------
        DeepSORTDetection
            A DeepSORT Detection that is wrapped around a darknet detection. 
        """
        return DeepSORTDetection([(darknetDetection.X-darknetDetection.Width/2),
                                  (darknetDetection.Y-darknetDetection.Height/2),
                                  darknetDetection.Height, darknetDetection.Height],
                                 float(darknetDetection.confidence), [], darknetDetection)

    def update_history(self, history: List[TrackedObject], new_detections: List[DarknetDetection], joblibbuffer: Optional[List[TrackedObject]] = None, db_connection: Optional[str] = None):
        """Update trajectory history with new detections.

        Parameters
        ----------
        history : List[TrackedObject]
            History list.
        new_detections : List[DarknetDetection]
            New detections from yolo.
        joblibbuffer : Optional[List[TrackedObject]], optional
            The joblib buffer, which will be saved at the end of runtime, by default None
        """
        wrapped_Detections = [self.make_detection_object(
            det) for det in new_detections]
        self._Tracker.predict()
        self._Tracker.update(wrapped_Detections)
        for track in self._Tracker.tracks:
            updated = False
            for to in history:
                if track.track_id == to.objID:
                    if track.time_since_update == 0:
                        # , k_velocity, k_acceleration)
                        to.update(track.darknetDet, track.mean)
                        if len(to.history) > self.historyDepth:
                            to.history.pop(0)
                    else:
                        # if arg in update is None, then time_since_update += 1
                        to.update()
                        if to.max_age <= to.time_since_update:
                            if joblibbuffer is not None:
                                joblibbuffer.append(to)
                            history.remove(to)
                    updated = True
                    break
            if not updated:
                newTrack = TrackedObject(
                    track.track_id, track.darknetDet, track._max_age)
                history.append(newTrack)
                if db_connection is not None:
                    logObject(db_connection, newTrack.objID, newTrack.label)


class TrajectoryNet:
    """
    A class representing the TrajectoryNet model for predicting trajectories.

    Parameters
    ----------
    model : str
        Path to the model file.
    debug : bool, optional
        Flag indicating whether to enable debug mode, by default False.

    Attributes
    ----------
    _logger : Logger
        Logger object.
    _model : OneVsRestClassifierWrapper
        Loaded model.

    Methods
    -------
    predict(feature_vector: np.ndarray) -> np.ndarray
        Predict trajectories.
    feature_extraction(trajectory: TrackedObject, feature_version: Literal["1", "7"]) -> np.ndarray
        Extract features from history.
    draw_clusters(cluster_centroids: np.ndarray, image: np.ndarray) -> np.ndarray
        Draw exit clusters on image.
    draw_prediction(trackedObject: TrackedObject, predicted_cluster: int, cluster_centers: np.ndarray, image: np.ndarray, color: Tuple = (0, 255, 0), thickness: int = 3) -> np.ndarray
        Draw predictions on image.
    draw_top_k_prediction(trackedObject: TrackedObject, predictions: np.ndarray, cluster_centers: np.ndarray, image: np.ndarray, k: int = 3, thickness: int = 3) -> np.ndarray
        Draw top k predictions on image.
    upscale_coordinate(pt1, pt2, shape: Tuple[int, int, int]) -> Tuple[int, int]
        Upscale coordinate from 0-1 range to image size.
    draw_history(trackedObject: TrackedObject, image: np.ndarray, color: Tuple = (0, 0, 255), thickness: int = 3) -> np.ndarray
        Draw trajectory history on image.

    """

    def __init__(self, model: str, debug: bool = False):
        self._logger = getLogger("TrajectoryNet_Logger")
        _logHandler = StreamHandler()
        if debug:
            self._logger.setLevel(DEBUG)
            _logHandler.setLevel(DEBUG)
        else:
            self._logger.setLevel(INFO)
            _logHandler.setLevel(INFO)
        _formatter = Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        _logHandler.setFormatter(_formatter)
        self._logger.addHandler(_logHandler)

        self._model: OneVsRestClassifierWrapper = load_model(model)
        self._logger.debug(f"Model: {self._model}")

    def predict(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Predict trajectories.

        Parameters
        ----------
        feature_vector : np.ndarray
            Feature vector.

        Returns
        -------
        np.ndarray
            Predicted trajectories.
        """
        return self._model.predict_proba(feature_vector)

    @staticmethod
    def feature_extraction(trajectory: TrackedObject, feature_version: Literal["1", "7"]) -> np.ndarray:
        """
        Extract features from history.

        Parameters
        ----------
        trajectory : TrackedObject
            Trajectory object.
        feature_version : Literal["1", "7"]
            Version of the feature extraction algorithm.

        Returns
        -------
        np.ndarray
            Extracted features.
        """
        if feature_version == "1":
            return trajectory.feature_v1()
        elif feature_version == "7":
            return trajectory.feature_v7()

    @staticmethod
    def draw_clusters(cluster_centroids: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Draw exit clusters on image.

        Parameters
        ----------
        cluster_centroids : np.ndarray
            Cluster x,y coordinates.
        image : np.ndarray
            Image to draw on.

        Returns
        -------
        np.ndarray
            Output image.
        """
        for i, cluster in enumerate(cluster_centroids):
            cv2.circle(image, (int(cluster[0]), int(
                cluster[1])), 10, (0, 0, 255), 3)
            cv2.putText(image, str(i), (int(cluster[0]), int(
                cluster[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    @staticmethod
    def draw_prediction(trackedObject: TrackedObject, predicted_cluster: int, cluster_centers: np.ndarray, image: np.ndarray, color: Tuple = (0, 255, 0), thickness: int = 3) -> np.ndarray:
        """
        Draw predictions on image.

        Parameters
        ----------
        trackedObject : TrackedObject
            Tracked object.
        predicted_cluster : int
            Index of the predicted cluster.
        cluster_centers : np.ndarray
            Cluster centers.
        image : np.ndarray
            Image to draw on.
        color : Tuple, optional
            Color of the line, by default (0, 255, 0).
        thickness : int, optional
            Thickness of the line, by default 3.

        Returns
        -------
        np.ndarray
            Output image.
        """
        cv2.line(image, (int(trackedObject.X), int(trackedObject.Y)), (int(
            cluster_centers[predicted_cluster][0]), int(cluster_centers[predicted_cluster][1])), color, thickness)
        cv2.putText(image, str(predicted_cluster), (int(trackedObject.X), int(
            trackedObject.Y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    @staticmethod
    def draw_top_k_prediction(trackedObject: TrackedObject, predictions: np.ndarray, cluster_centers: np.ndarray, image: np.ndarray, k: int = 3, thickness: int = 3) -> np.ndarray:
        """
        Draw top k predictions on image.

        Parameters
        ----------
        trackedObject : TrackedObject
            Tracked object.
        predictions : np.ndarray
            Predictions.
        cluster_centers : np.ndarray
            Cluster centers.
        image : np.ndarray
            Image to draw on.
        k : int, optional
            Number of top predictions to draw, by default 3.
        thickness : int, optional
            Thickness of the line, by default 3.

        Returns
        -------
        np.ndarray
            Output image.
        """
        top_k = np.argsort(predictions)[-k:]
        for i in top_k:
            if i == top_k[-1]:
                cv2.line(image, (int(trackedObject.X), int(trackedObject.Y)), (int(
                    cluster_centers[i][0]), int(cluster_centers[i][1])), (0, 255, 0), thickness)
                cv2.putText(image, str(top_k[-1]), (int(trackedObject.X), int(
                    trackedObject.Y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.line(image, (int(trackedObject.X), int(trackedObject.Y)), (int(
                    cluster_centers[i][0]), int(cluster_centers[i][1])), (0, 0, 255), thickness)

    @staticmethod
    @lru_cache(maxsize=128)
    def upscale_coordinate(pt1, pt2, shape: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Upscale coordinate from 0-1 range to image size.

        Parameters
        ----------
        pt1 : float
            Coordinate x.
        pt2 : float
            Coordinate y.
        shape : Tuple[int, int, int]
            Image shape.

        Returns
        -------
        Tuple[int, int]
            Upscaled coordinate.
        """
        aspect_ratio = shape[1]/shape[0]
        return (pt1*shape[1])/aspect_ratio, pt2*shape[0]

    @staticmethod
    def draw_history(trackedObject: TrackedObject, image: np.ndarray, color: Tuple = (0, 0, 255), thickness: int = 3) -> np.ndarray:
        """
        Draw trajectory history on image.

        Parameters
        ----------
        trackedObject : TrackedObject
            Tracked object.
        image : np.ndarray
            Image to draw on.
        color : Tuple, optional
            Color of the line, by default (0, 0, 255).
        thickness : int, optional
            Thickness of the line, by default 3.

        Returns
        -------
        np.ndarray
            Output image.
        """
        for i in range(len(trackedObject.history)-1):
            cv2.line(image, (int(trackedObject.history[i].X), int(trackedObject.history[i].Y)), (int(
                trackedObject.history[i+1].X), int(trackedObject.history[i+1].Y)), thickness=thickness, color=color)

    @staticmethod
    def draw_velocity_vector(trackedObject: TrackedObject, image: np.ndarray, color: Tuple = (0, 0, 255), thickness: int = 3) -> np.ndarray:
        """
        Draw velocity vector on image.

        Parameters
        ----------
        trackedObject : TrackedObject
            Tracked object.
        image : np.ndarray
            Image to draw on.
        color : Tuple, optional
            Color of the line, by default (0, 0, 255).
        thickness : int, optional
            Thickness of the line, by default 3.

        Returns
        -------
        np.ndarray
            Output image.
        """
        # if trackedObject.history_X.shape[0] >= 5:
        # vx = savgol_filter(trackedObject.history_X, 5, 2, 1)[-1] * 2
        # vy = savgol_filter(trackedObject.history_Y, 5, 2, 1)[-1] * 2
        cv2.line(image, (int(trackedObject.X), int(trackedObject.Y)), (int(
            trackedObject.X + trackedObject.VX * 3), int(trackedObject.Y + trackedObject.VY * 3)), thickness=thickness, color=color)


class Detector:
    """Detection pipeline class.
    This class is used to run the detection pipeline.

    Parameters
    ----------
    source : str
        Video source path.
    outdir : str
        Output directory path. 
    model : str
        Path to model weights file.
    database : bool, optional
        Save results to database, by default False
    joblib : bool, optional
        Save results to joblib, by default False
    debug : bool, optional
        Debug flag, by default True

    Attributes
    ----------
    _source : Path
        Video source path.
    _outdir : Path
        Output directory path.
    _model : Model
        Loaded scikit-learn model.
    _dataset : LoadImages
        LoadImages object, which is used to load images from video source. 
    _database : Path
        Path to database file.
    _joblib : Path
        Path to joblib file.
    _joblibbuffer : List
        Joblib buffer, which is used to store TrackedObject objects.
    _history : List
        History list, which is used to store TrackedObject objects.

    Methods
    -------
    _init_logger(debug: bool = False) -> None
        Init logger.
    _init_output_directory(path: Optional[str] = None) -> None
        Init output directory.
    _init_video_writer() -> None
        Init video capture.
    generate_db_path(source: Union[str, Path], outdir: Optional[Union[str, Path]] = None, suffix: str = ".joblib", logger: Optional[Logger] = None) -> Path
        Generate output path name from source and output directory path.
    filter_objects(new_detections: Union[List, torch.Tensor], frame_number: int, names: List[str] = ["car"]) -> List[DarknetDetection]
        Filter out detections that are not in the names list.
    run(yolo: Yolov7, deepSort: Optional[DeepSORT] = None, trajectoryNet: TrajectoryNet = None, show: bool = False)
        Run detection pipeline.

    Examples
    --------
    >>> from DetectionPipeline import Detector, Yolov7, DeepSORT
    >>> detector = Detector(source="path/to/video.mp4", outdir="path/to/output/directory", database=True, joblib=True) # database and joblib are optional
    >>> yolo = Yolov7(weights="path/to/model/weights.pt", conf_thres=0.5, iou_thres=0.5, half=True, device="cuda", debug=True)
    >>> deepSort = DeepSORT(max_age=30, debug=True)
    >>> detector.run(yolo, deepSort, show=True)
    """

    def __init__(self, source: str, outdir: Optional[str] = None, database: bool = False, joblib: bool = False, debug: bool = True):
        self._init_logger(debug=debug)
        self._source = Path(source)
        self._init_output_directory(path=outdir)
        self._init_video_writer()
        self._dataset = LoadImages(self._source, img_size=640, stride=32)
        self._logger.debug(f"Files: {self._dataset.files}")
        if database:
            self._databases = [self.generate_db_path(f, self._outdir, suffix=".db", logger=self._logger)
                               for f in self._dataset.files]
        else:
            self._databases = None
        if joblib:
            self._joblibs = [self.generate_db_path(f, self._outdir, suffix=".joblib", logger=self._logger)
                             for f in self._dataset.files]
            self._joblibbuffers = [[] for _ in self._dataset.files]
        else:
            self._joblibbuffers = [None] * len(self._dataset.files)
        self._logger.debug(f"Joblib buffers: {self._joblibbuffers}")
        self._history = []

    def _init_logger(self, debug: bool = False) -> None:
        """Init logger.

        Parameters
        ----------
        debug : bool, optional
            Debug flag, by default False

        Returns
        -------
        None
        """
        self._logger = getLogger("Pipeline_Logger")
        _logHandler = StreamHandler()
        if debug:
            self._logger.setLevel(DEBUG)
            _logHandler.setLevel(DEBUG)
        else:
            self._logger.setLevel(INFO)
            _logHandler.setLevel(INFO)
        _formatter = Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        _logHandler.setFormatter(_formatter)
        self._logger.addHandler(_logHandler)

    def _init_output_directory(self, path: Optional[str] = None) -> None:
        """Init output directory.

        Parameters
        ----------
        path : Path
            Path to output directory.

        Returns
        -------
        None
        """
        if path is not None:
            self._outdir = Path(path)
        else:
            self._outdir = self._source.parent
        if not self._outdir.exists():
            self._outdir.mkdir(parents=True)
        self._record_path = self._outdir.joinpath("runs")
        if not self._record_path.exists():
            self._record_path.mkdir(parents=True)
        self._logger.debug(f"Output directory: {path}")
        self._logger.debug(f"Record path: {self._record_path}")

    def _init_video_writer(self) -> None:
        """Init video capture.

        Returns
        -------
        cv2.VideoCapture
            Video capture object.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        n_runs = len(list(self._record_path.glob("*.mp4")))
        # self._capture = cv2.VideoWriter(filename=self._record_path.joinpath("run_{}.mp4".format(n_runs)), fourcc=fourcc, fps=30)

    @staticmethod
    def generate_db_path(source: Union[str, Path], outdir: Optional[Union[str, Path]] = None, suffix: str = ".joblib", logger: Optional[Logger] = None) -> Path:
        """Generate output path name from source and output directory path.

        Parameters
        ----------
        source : Union[str, Path]
            Video source path.
        outdir : Optional[Union[str, Path]], optional
            Output directory path, if none use source directory path as output directory path, by default None
        suffix : str
            The db suffix, eg. .joblib, .db etc...

        Returns
        -------
        Path
            The path to output.
        """
        _source = Path(source)
        if outdir is None:
            out = _source.with_suffix(suffix=suffix)
            if logger is not None:
                logger.debug(f"Output path: {out}")
            return out
        out = Path(outdir).joinpath(_source.with_suffix(suffix=suffix).name)
        if not Path(outdir).exists():
            Path(outdir).mkdir(parents=True)
        if logger is not None:
            logger.debug(f"Output path: {out}")
        return out

    @staticmethod
    def filter_objects(new_detections: Union[List, torch.Tensor], frame_number: int, names: List[str] = ["car"]) -> List[DarknetDetection]:
        """Filter out detections that are not in the names list.

        Parameters
        ----------
        new_detections : List
            New detections from yolo
        frame_number : int
            Frame number
        names : List[str], optional
            Names to include, by default ["car"]

        Returns
        -------
        List[DarknetDetection]
            List of Detection objects
        """
        targets = []
        for label, conf, bbox in new_detections:
            # bbox: x, y, w, h
            if label in names:
                targets.append(DarknetDetection(
                    label, conf, bbox[0], bbox[1], bbox[2], bbox[3], frame_number))
        return targets

    def run(self, yolo: Yolov7, deepSort: Optional[DeepSORT] = None, trajectoryNet: TrajectoryNet = None, show: bool = False, feature_version: Literal["1", "7"] = "7", k: int = 1):
        """Run detection pipeline.

        Parameters
        ----------
        yolo : Yolov7
            Yolov7 object
        deepSort: Optional[DeepSORT], optional
            DeepSORT object, by default None
        trajectoryNet : Optional[TrajectoryNet], optional
            TrajectoryNet object, by default None
        show : bool, optional
            Show flag to visualize frames with cv2 GUI, by default False
        """
        # mask = masker(img) # mask out not wanted areas
        yolo.warmup()  # warm up yolo model
        # previous_path = None
        _, _, im0s, _ = next(iter(self._dataset))
        cluster_centroids = np.array([trajectoryNet.upscale_coordinate(
            coord[0], coord[1], im0s.shape) for coord in trajectoryNet._model.cluster_centroids])
        pooled_mask = trajectoryNet._model.pooled_classes
        self._logger.debug(f"Cluster centroids: {cluster_centroids}")
        for path, img, im0s, vid_cap in self._dataset:
            p, s, im0, frame = path, '', im0s.copy(), getattr(self._dataset, 'frame', 0)
            self._logger.debug(f"Input image shape: {img.shape}")
            # run inference
            preds = yolo.infer(img)
            # postprocess predictions
            preds = yolo.postprocess(preds, im0, img, show=show)
            self._logger.debug(f"Detections: {preds}")
            # filter out unwanted detections and create Detection objects
            new_detections = self.filter_objects(
                new_detections=preds, frame_number=frame)
            self._logger.debug(
                f"New detections: {[d.label for d in new_detections]}")
            # update tracker and history
            deepSort.update_history(history=self._history, new_detections=new_detections,
                                    joblibbuffer=self._joblibbuffers[self._dataset.count], db_connection=self._databases)
            self._logger.debug(f"History: {self._history}")
            if trajectoryNet is not None:
                # draw clusters
                trajectoryNet.draw_clusters(
                    cluster_centroids=cluster_centroids, image=im0)
                for t in self._history:
                    # extract features from history
                    feature_vector = TrackedObject.downscale_feature(
                        trajectoryNet.feature_extraction(t, feature_version=feature_version))
                    self._logger.debug(f"Feature vectors: {feature_vector}")
                    # predict trajectories
                    if feature_vector is not None:
                        predictions = trajectoryNet.predict(
                            feature_vector.reshape(1, -1))
                        self._logger.debug(f"Predictions: {predictions}")
                        # pool predictions
                        # predictions = mask_predictions(
                        #     predictions, pooled_mask)
                        # self._logger.debug(f"Predictions: {predictions}")
                        # draw predictions
                        trajectoryNet.draw_top_k_prediction(
                            trackedObject=t, predictions=predictions[0], cluster_centers=cluster_centroids, image=im0, k=k)
                    trajectoryNet.draw_history(t, im0, thickness=1)
                    trajectoryNet.draw_velocity_vector(
                        t, im0, color=(255, 255, 255))
            if show:
                cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('p'):
                cv2.waitKey(0)
        for i, buf in enumerate(self._joblibbuffers):
            self._logger.debug(
                f"Joblib buffer {self._joblibs[i]}: len({len(buf)})")
            dump(buf, self._joblibs[i])


if __name__ == "__main__":
    yolo = Yolov7(
        weights="/media/pecneb/970evoplus/gitclones/computer_vision_research/computer_vision_research/yolov7/yolov7.pt", debug=True)
    deepSort = DeepSORT(debug=True)
    det = Detector(source="/media/pecneb/DataStorage/computer_vision_research_test_videos/test_videos/short_ones/",  # Bellevue_116th_NE12th__2017-09-11_11-08-33.mp4",
                   outdir="./research_data/short_test_videos/", database=False, joblib=True, debug=True)
    # model="/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/models/SVM_7.joblib")
    det.run(yolo=yolo, deepSort=deepSort, show=True, k=3)
