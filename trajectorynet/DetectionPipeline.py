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
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from joblib import dump
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection as DeepSORTDetection
from deep_sort.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.deep_sort.tracker import Tracker
from torch import nn
from yolov7.models.common import Conv
from yolov7.models.experimental import Ensemble
from yolov7.utils.datasets import LoadImages, letterbox
from yolov7.utils.general import (check_img_size, check_imshow,
                                  non_max_suppression, scale_coords, xyxy2xywh)
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, time_synchronized

from dataManagementClasses import Detection as DarknetDetection
from dataManagementClasses import TrackedObject
from masker import masker
from utility.databaseLogger import logObject


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
    history_length : int, optional
        Length of history, the detections to hold in buffer, by default 30
    output : Optional[str], optional
        Output directory, by default None
    max_cosine_distance : float, optional
        Gating threshold for cosine distance metric (object appearance), by default 10.0
    max_iou_distance : float, optional
        Max intersection over union distance, by default 0.7
    nn_budget : float, optional
        Maximum size of the appearence descriptor gallery, by default 100

    Attributes
    ----------
    source : Union[int, str]
        Video source, can be webcam number, video path, directory path containing videos
    history_length : int
        Length of history, the detections to hold in buffer
    output : Optional[str]
        Output directory
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance)
    max_iou_distance : float
        Max intersection over union distance       
    nn_budget : float
        Maximum size of the appearence descriptor gallery
    historyDepth : int
        Length of history 

    _history : List

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


class Detector:
    """Detection pipeline class.
    This class is used to run the detection pipeline.

    Parameters
    ----------
    source : str
        Video source path.
    outdir : str
        Output directory path. 

    Attributes
    ----------
    _source : Path
        Video source path.
    _outdir : Path
        Output directory path.

    Methods
    -------
    generate_db_path(source: Union[str, Path], outdir: Optional[Union[str, Path]] = None, suffix: str = ".joblib", logger: Optional[Logger] = None) -> Path
        Generate output path name from source and output directory path.
    filter_objects(new_detections: Union[List, torch.Tensor], frame_number: int, names: List[str] = ["car"]) -> List[DarknetDetection]
        Filter out detections that are not in the names list.
    run(yolo: Yolov7, deepSort: Optional[DeepSORT] = None, show: bool = False)
        Run detection pipeline.
    """

    def __init__(self, source: str, outdir: Optional[str] = None, database: bool = False, joblib: bool = False, debug: bool = True):
        # region init logger
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
        # endregion

        self._source = Path(source)
        self._logger.debug(f"Video source: {self._source}")
        if outdir is not None:
            self._outdir = Path(outdir).absolute()
            self._logger.debug(f"Output directory: {self._outdir}")
        else:
            self._outdir = self._source.parent

        self._dataset = LoadImages(self._source, img_size=640, stride=32)
        self._database = None
        self._joblib = None

        if database:
            self._database = self.generate_db_path(
                self._source, outdir=self._outdir, suffix=".db", logger=self._logger)
        if joblib:
            self._joblib = self.generate_db_path(
                self._source, outdir=self._outdir, suffix=".joblib", logger=self._logger)
        self._joblibbuffer = []
        self._history = []

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

    def run(self, yolo: Yolov7, deepSort: Optional[DeepSORT] = None, show: bool = False):
        """Run detection pipeline.

        Parameters
        ----------
        yolo : Yolov7
            Yolov7 object
        deepSort: Optional[DeepSORT], optional
            DeepSORT object, by default None
        show : bool, optional
            Show flag to visualize frames with cv2 GUI, by default False
        """
        # mask = masker(img) # mask out not wanted areas
        yolo.warmup()
        for path, img, im0s, vid_cap in self._dataset:
            p, s, im0, frame = path, '', im0s.copy(), getattr(self._dataset, 'frame', 0)
            self._logger.debug(f"Input image shape: {img.shape}")
            preds = yolo.infer(img)
            preds = yolo.postprocess(preds, im0, img, show=show)
            self._logger.debug(f"Predictions: {preds}")
            new_detections = self.filter_objects(
                new_detections=preds, frame_number=frame)
            self._logger.debug(
                f"New detections: {[d.label for d in new_detections]}")
            deepSort.update_history(history=self._history, new_detections=new_detections,
                                    joblibbuffer=self._joblibbuffer, db_connection=self._database)
            if show:
                cv2.imshow(p, im0)
            if cv2.waitKey(1) == ord('q'):
                break
        if self._joblib is not None:
            self._logger.debug(f"Saving joblib buffer to {self._joblib}.")
            self._logger.debug(f"Length of buffer: {len(self._joblibbuffer)}")
            t0 = time_synchronized() 
            dump(value=self._joblibbuffer, filename=self._joblib)
            t1 = time_synchronized()
            self._logger.debug(f"Joblib dump time: {t1-t0}s")


if __name__ == "__main__":
    yolo = Yolov7(
        weights="/media/pecneb/970evoplus/gitclones/computer_vision_research/computer_vision_research/yolov7/yolov7.pt", debug=True)
    deepSort = DeepSORT(debug=True)
    det = Detector(source="/media/pecneb/DataStorage/computer_vision_research_test_videos/test_videos/rouen_video.avi",
                   outdir="./research_data/rouren_video/", database=False, joblib=True, debug=True)
    det.run(yolo=yolo, deepSort=deepSort, show=True)
