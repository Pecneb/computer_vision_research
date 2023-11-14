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

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import tqdm
from utility import databaseLoader
from utility.featurevector import FeatureVector


@dataclass
class Detection:
   """Class for storing detections of YOLO Darknet.

    Parameters
    ----------
    label : str
        Classification name ex. car, person, bicycle, etc.
    confidence : float
        The likeliness that the detected object is really the above label, 0.0-1.0.
    X : float
        Center of bounding box in the X axis.
    Y : float
        Center of bounding box in the Y axis.
    Width : float
        Width of the bounding box.
    Height : float
        Height of the bounding box.
    frameID : int
        The number of the frame, the detection is acquired.
    VX : float, optional
        Velocity in the X axis. Used in the analyzer script and databaseLoader script.
    VY : float, optional
        Velocity in the Y axis. Used in the analyzer script and databaseLoader script.
    AX : float, optional
        Acceleration in the X axis. Used in the analyzer script and databaseLoader script.
    AY : float, optional
        Acceleration in the Y axis. Used in the analyzer script and databaseLoader script.
    objID : int, optional
        Object ID. Used in the analyzer script and databaseLoader script.

    Attributes
    ----------
    label : str
        Classification name ex. car, person, bicycle, etc.
    confidence : float
        The likeliness that the detected object is really the above label, 0.0-1.0.
    X : float
        Center of bounding box in the X axis.
    Y : float
        Center of bounding box in the Y axis.
    Width : float
        Width of the bounding box.
    Height : float
        Height of the bounding box.
    frameID : int
        The number of the frame, the detection is acquired.
    VX : float, optional
        Velocity in the X axis. Used in the analyzer script and databaseLoader script.
    VY : float, optional
        Velocity in the Y axis. Used in the analyzer script and databaseLoader script.
    AX : float, optional
        Acceleration in the X axis. Used in the analyzer script and databaseLoader script.
    AY : float, optional
        Acceleration in the Y axis. Used in the analyzer script and databaseLoader script.
    objID : int, optional
        Object ID. Used in the analyzer script and databaseLoader script.

    Methods
    -------
    __repr__() -> str
        Returns a string representation of the Detection object.
    __eq__(other) -> bool
        Compares two Detection objects for equality.

    """
    label: str
    confidence: float
    X: float
    Y: float
    Width: float
    Height: float
    frameID: int
    VX: float = field(init=False,)
    VY: float = field(init=False)
    AX: float = field(init=False)
    AY: float = field(init=False)
    objID: int = field(init=False)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Detection object.

        Returns
        -------
        str
            A string representation of the Detection object.
        """
        return f"Label: {self.label}, Confidence: {self.confidence}, X: {self.X}, Y: {self.Y}, Width: {self.Width}, Height: {self.Height}, Framenumber: {self.frameID}"

    def __eq__(self, other) -> bool:
        """
        Returns True if the given Detection object is equal to this Detection object, False otherwise.

        Parameters
        ----------
        other : Detection
            The Detection object to compare with.

        Returns
        -------
        bool
            True if the given Detection object is equal to this Detection object, False otherwise.
        """
        if self.label != other.label:
            return False
        if self.confidence != other.confidence:
            return False
        if self.X != other.X:
            return False
        if self.Y != other.Y:
            return False
        if self.Width != other.Width:
            return False
        if self.Height != other.Height:
            return False
        if self.frameID != other.frameID:
            return False
        return True


@dataclass
class TrackedObject():
    """
    A class representing a tracked object in a video.

    Attributes
    ----------
    objID : int
        The ID of the object.
    label : int
        The label of the object.
    futureX : list
        A list of future X positions of the object.
    futureY : list
        A list of future Y positions of the object.
    history : List[Detection]
        A list of Detection objects representing the history of the object.
    history_X : np.ndarray
        An array of X positions of the object in the history.
    history_Y : np.ndarray
        An array of Y positions of the object in the history.
    history_VX_calculated : np.ndarray
        An array of calculated X velocities of the object in the history.
    history_VY_calculated : np.ndarray
        An array of calculated Y velocities of the object in the history.
    history_AX_calculated : np.ndarray
        An array of calculated X accelerations of the object in the history.
    history_AY_calculated : np.ndarray
        An array of calculated Y accelerations of the object in the history.
    isMoving : bool
        A boolean indicating whether the object is moving or not.
    time_since_update : int
        The time elapsed since the last update of the object.
    max_age : int
        The maximum age of the object.
    mean : list
        A list of mean values of the object.
    X : int
        The X position of the object.
    Y : int
        The Y position of the object.
    VX : float
        The X velocity of the object.
    VY : float
        The Y velocity of the object.
    AX : float
        The X acceleration of the object.
    AY : float
        The Y acceleration of the object.
    _dataset : str
        The dataset of the object.

    Methods
    -------
    __init__(self, id: int, first: Detection, max_age: int = 30)
        Initializes a TrackedObject instance.
    __repr__(self) -> str
        Returns a string representation of the TrackedObject instance.
    __hash__(self) -> int
        Returns a hash value of the TrackedObject instance.
    __eq__(self, other) -> bool
        Returns True if the TrackedObject instance is equal to the other TrackedObject instance.
    avgArea(self) -> np.ndarray
        Calculates the average area of the bounding boxes of the object.
    updateAccel(self, new_vx, old_vx, new_vy, old_vy) -> None
        Calculates acceleration based on Kalman filter's velocity.
    upscale_feature(featureVector: np.ndarray, framewidth: int = 1920, frameheight: int = 1080) -> np.ndarray
        Rescales normalized coordinates with the given frame sizes.
    downscale_feature(featureVector: np.ndarray, framewidth: int = 1920, frameheight: int = 1080) -> np.ndarray
        Normalizes coordinates with the given frame sizes.
    feature_v1(self) -> Optional[np.ndarray]
        Extracts version 1 feature vector from history.
    feature_v1_SG(self, window_length: int = 7, polyorder: int = 2) -> Optional[np.ndarray]
        Extracts version 1SG feature vector from history.
    feature_v3(self) -> np.ndarray
        Returns version 3 feature vector.
    feature_v7(self, history_size: int = 30, weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]
        Extracts feature version 7 from history.
    feature_v7_SG(self, history_size: int = 30, weights: np.ndarray = None, window_length: int = 7, polyorder: int = 2) -> Optional[np.ndarray]
        Extracts feature version 7SG from history.
    feature_v8(self, history_size: int = 30) -> Optional[np.ndarray]
        Extracts feature version 8 from history.
    feature_v8_SG(self, history_size: int = 30, window_length: int = 7, polyorder: int = 2) -> Optional[np.ndarray]
        Extracts feature version 8SG from history.
    feature_v5(self, n_weights: int) -> Optional[np.ndarray]
        Returns version 5 feature vector.
    feature_vector(self, version: str = '1', **kwargs) -> Optional[np.ndarray]
        Returns the corresponding feature vector version of the given version argument.

    """
    objID: int
    label: int = field(init=False)
    futureX: list = field(init=False)
    futureY: list = field(init=False)
    history: List[Detection] = field(init=False)
    history_X: np.ndarray = field(init=False)
    history_Y: np.ndarray = field(init=False)
    history_VX_calculated: np.ndarray = field(init=False)
    history_VY_calculated: np.ndarray = field(init=False)
    history_AX_calculated: np.ndarray = field(init=False)
    history_AY_calculated: np.ndarray = field(init=False)
    isMoving: bool = field(init=False)
    time_since_update: int = field(init=False)
    max_age: int
    mean: list = field(init=False)
    X: int
    Y: int
    VX: float = field(init=False)
    VY: float = field(init=False)
    AX: float = field(init=False)
    AY: float = field(init=False)
    # bugged: int = field(init=False)
    # featureVector: np.ndarray = field(init=False)
    _dataset: str = field(init=False)

    def __init__(self, id: int, first: Detection, max_age: int = 30):
        self.objID = id
        self.history = [first]
        self.history_X = np.array([first.X])
        self.history_Y = np.array([first.Y])
        self.history_VX_calculated = np.array([])
        self.history_VY_calculated = np.array([])
        self.history_VT = np.array([])
        self.history_AX_calculated = np.array([])
        self.history_AY_calculated = np.array([])
        self.X = first.X
        self.Y = first.Y
        self.VX = 0
        self.VY = 0
        self.AX = 0
        self.AY = 0
        self.history[-1].VX = self.VX
        self.history[-1].VY = self.VY
        self.history[-1].AX = self.AX
        self.history[-1].AY = self.AY
        self.label = first.label
        self.isMoving = False
        self.futureX = []
        self.futureY = []
        self.max_age = max_age
        self.time_since_update = 0
        self.mean = []
        # self.bugged = 0
        self._dataset = ""

    def __repr__(self) -> str:
        return f"ID: {self.objID}, Label: {self.label}, X: {self.X}, Y: {self.Y}, VX: {self.VX}, VY: {self.VY}, AX: {self.AX}, AY: {self.AY}, Time since update: {self.time_since_update}, Max age: {self.max_age}, Dataset: {self._dataset}"

    def __hash__(self) -> int:
        retval = int(
            self.objID+np.sum([self.history[i].frameID for i in range(len(self.history))]))
        # print(retval, self.objID, self._dataset)
        return retval

    def __eq__(self, other) -> bool:
        for i in range(len(self.history)):
            for j in range(len(other.history)):
                if self.history[i] != other.history[j]:
                    return False
        return self.objID == other.objID

    def avgArea(self) -> np.ndarray:
        """Calculate the average area of the bounding boxes of the object.

        Returns
        -------
        float
            The average area of the bounding boxes of the object.
        """
        areas = [(det.Width*det.Height) for det in self.history]
        return np.average(areas)

    def updateAccel(self, new_vx, old_vx, new_vy, old_vy) -> None:
        """Calculate acceleration based on Kalman filter's velocity.

        Calculates acceleration using the formula:
        (new_v - old_v) / (new_time - old_time)
        where the time between the two detections is (new_time-old_time).

        Parameters
        ----------
        new_vx : float
            The new velocity of x.
        old_vx : float
            The old velocity of x from the previous detection.
        new_vy : float
            The new velocity of y.
        old_vy : float
            The old velocity of y from the previous detection.

        Returns
        -------
        None
        """
        self.AX = (new_vx - old_vx) / (self.time_since_update+1)
        self.AY = (new_vy - old_vy) / (self.time_since_update+1)

    @staticmethod
    def upscale_feature(featureVector: np.ndarray, framewidth: int = 1920, frameheight: int = 1080) -> np.ndarray:
        """Rescale normalized coordinates with the given frame sizes.

        Parameters
        ----------
        featureVector : ndarray
            Feature vector of the track
        framewidth : int, optional
            Frame width, by default 1920
        frameheight : int, optional
            Frame height, by default 1080

        Returns
        -------
        ndarray
            Upscaled feature vector
        """
        if featureVector is None:
            return None
        ratio = framewidth / frameheight

        ret_featureVector = np.array([])
        for i in range(featureVector.shape[0]):
            if i == 0 or i % 2 == 0:
                ret_featureVector = np.append(
                    ret_featureVector, [featureVector[i]*framewidth/ratio])
            else:
                ret_featureVector = np.append(
                    ret_featureVector, [frameheight*featureVector[i]])
        return ret_featureVector

    @staticmethod
    def downscale_feature(featureVector: np.ndarray, framewidth: int = 1920, frameheight: int = 1080) -> np.ndarray:
        """Normalize coordinates with the given frame sizes.
        Normalization is done by dividing the coordinates with the frame sizes,
        but the X coordinates are divided with the ratio of the frame width and height.

        Parameters
        ----------
        featureVector : ndarray
            Feature vector of the track
        framewidth : int, optional
            Frame width, by default 1920
        frameheight : int, optional
            Frame height, by default 1080

        Returns
        -------
        ndarray
            Normalized feature vector
        """
        if featureVector is None:
            return None
        ratio = framewidth / frameheight

        ret_featureVector = np.array([])
        for i in range(featureVector.shape[0]):
            if i == 0 or i % 2 == 0:
                ret_featureVector = np.append(
                    ret_featureVector, [featureVector[i]/framewidth*ratio])
            else:
                ret_featureVector = np.append(
                    ret_featureVector, [featureVector[i]/frameheight])
        return ret_featureVector

    def feature_v1(self) -> Optional[np.ndarray]:
        """Extract version 1 feature vector from history.

        Returns
        -------
        ndarray
            Feature vector
        """
        n = len(self.history)-1
        if n < 3:
            return None
        return FeatureVector._1(self.history)

    def feature_v1_SG(self, window_length: int = 7, polyorder: int = 2) -> Optional[np.ndarray]:
        """Extract version 1SG feature vector from history.

        Returns
        -------
        ndarray
            Feature vector
        """
        n = len(self.history)-1
        if n < 3:
            return None
        return FeatureVector._1_SG(self.history, window_length=window_length, polyorder=polyorder)

    def feature_v3(self) -> np.ndarray:
        """Return version 3 feature vector.

        Returns
        -------
        ndarray 
            Feature vector version 3
        """
        n = len(self.history)-1
        if n < 3:
            return None
        return np.array([self.history[0].X, self.history[0].Y,
                        self.history[n//2].X, self.history[n//2].Y,
                        self.history[n].X, self.history[n].Y])

    def feature_v7(self, history_size: int = 30, weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract feature version 7 from history.

        Parameters
        ----------
        history_size : int, optional
            Size of the history that should be used for feature vector creation, by default 30

        Returns
        -------
        Optional[np.ndarray]
            Feature vector or None, if history is not yet the size of history_size, then return None
        """
        if self.history_X.shape[0] < history_size:
            return None
        return FeatureVector._7(
            x=self.history_X[-history_size:],
            y=self.history_Y[-history_size:],
            vx=self.history_VX_calculated[-history_size:],
            vy=self.history_VY_calculated[-history_size:],
            weights=weights,
        )

    def feature_v7_SG(self, history_size: int = 30, weights: np.ndarray = None, window_length: int = 7, polyorder: int = 2) -> Optional[np.ndarray]:
        """Extract feature version 7SG from history.

        Parameters
        ----------
        history_size : int, optional
            Size of the history that should be used for feature vector creation, by default 30
        window_length : int, optional
            The size of the Savitzky Golay filter, by default 7
        polyorder : int, optional
            The polynomial order of the Savitzky Golay filter, by default 7

        Returns
        -------
        Optional[np.ndarray]
            Feature vector or None, if history is not yet the size of history_size, then return None
        """
        if self.history_X.shape[0] < history_size:
            return None
        return FeatureVector._7_SG(
            x=self.history_X[-history_size:],
            y=self.history_Y[-history_size:],
            weights=weights,
            window_length=window_length,
            polyorder=polyorder
        )

    def feature_v8(self, history_size: int = 30) -> Optional[np.ndarray]:
        """Extract feature version 7 from history.

        Parameters
        ----------
        history_size : int, optional
            Size of the history that should be used for feature vector creation, by default 30

        Returns
        -------
        Optional[np.ndarray]
            Feature vector or None, if history is not yet the size of history_size, then return None
        """
        if self.history_X.shape[0] < history_size:
            return None
        return FeatureVector._8(
            x=self.history_X[-history_size:],
            y=self.history_Y[-history_size:]
        )

    def feature_v8_SG(self, history_size: int = 30, window_length: int = 7, polyorder: int = 2) -> Optional[np.ndarray]:
        """Extract feature version 7SG from history.

        Parameters
        ----------
        history_size : int, optional
            Size of the history that should be used for feature vector creation, by default 30
        window_length : int, optional
            The size of the Savitzky Golay filter, by default 7
        polyorder : int, optional
            The polynomial order of the Savitzky Golay filter, by default 7

        Returns
        -------
        Optional[np.ndarray]
            Feature vector or None, if history is not yet the size of history_size, then return None
        """
        if self.history_X.shape[0] < history_size:
            return None
        return FeatureVector._8_SG(
            x=self.history_X[-history_size:],
            y=self.history_Y[-history_size:],
            window_length=window_length,
            polyorder=polyorder
        )

    def feature_v5(self, n_weights: int) -> Optional[np.ndarray]:
        """Return version 5 feature vector.

        This feature vector consists of the first and last detection's X and Y coordinates.
        The n_weight number controls the number of inserted coordinates between the first and the last detection's coordinates.

        Parameters
        ----------
        n_weights : int
            The number of inserted coordinates between the first and the last detection's coordinates.

        Returns
        -------
        Optional[np.ndarray]
            A numpy ndarray or None if the number of inserted coordinates is less than 1.
        """
        n = (self.history_X.shape[0] // 2)
        if (n // n_weights) < 1:
            return None
        feature = np.array([self.history_X[0], self.history_Y[0],
                            self.history_X[-1], self.history_Y[-1]])
        feature = insert_weights_into_feature_vector(
            n, self.history_X.shape[0], n_weights, self.history_X, self.history_Y, 2, feature)
        return feature

    def feature_vector(self, version: str = '1', **kwargs) -> Optional[np.ndarray]:
        """Return the corresponding feature vector version of the given version argument.
        Pass additional keyword arguments.

        Parameters
        ----------
        version : str, optional
            Version string of the feature vector, by default '1'

        Returns
        -------
        np.ndarray
            Feature vector
        """
        if version == '1':
            return self.feature_v1()
        elif version == '7':
            return self.feature_v7(history_size=kwargs["history_size"])
        elif version == '7SG':
            return self.feature_v7_SG(history_size=kwargs["history_size"], window_length=kwargs["window_length"], polyorder=kwargs["polyorder"])
        elif version == '8':
            return self.feature_v8(history_size=kwargs["history_size"])
        elif version == '8SG':
            return self.feature_v8_SG(history_size=kwargs["history_size"], window_length=kwargs["window_length"], polyorder=kwargs["polyorder"])
        return None

    def update_velocity(self, k: int = 10):
        """Calculate velocity from X,Y coordinates.
        """
        raise DeprecationWarning("This method is deprecated!")
        if self.history_X.shape[0] < k:
            self.history_VX_calculated = np.append(
                self.history_VX_calculated, [0])
            self.history_VY_calculated = np.append(
                self.history_VY_calculated, [0])
        else:
            # self.history_DT = np.append(self.history_DT, [self.history[-1].frameID - self.history[-k].frameID])
            # if self.history_DT[-1] == 0:
            #    dx = 0
            #    dy = 0
            # else:
            dx = (self.history_X[-1] - self.history_X[-k]) / \
                (self.history[-1].frameID - self.history[-k].frameID)
            dy = (self.history_Y[-1] - self.history_Y[-k]) / \
                (self.history[-1].frameID - self.history[-k].frameID)
            self.history_VX_calculated = np.append(
                self.history_VX_calculated, [dx])
            self.history_VY_calculated = np.append(
                self.history_VY_calculated, [dy])
        self.history_VT = np.append(
            self.history_VT, [self.history[-1].frameID])

    def update_accel(self, k: int = 2):
        """Calculate velocity from X,Y coordinates.
        """
        raise DeprecationWarning("This method is deprecated!")
        if self.history_VX_calculated.shape[0] < k:
            self.history_AX_calculated = np.append(
                self.history_AX_calculated, [0])
            self.history_AY_calculated = np.append(
                self.history_AY_calculated, [0])
        else:
            dt = (self.history_VT[-1] - self.history_VT[-k])
            if dt == 0:
                dvx = 0
                dvy = 0
            else:
                dvx = (
                    self.history_VX_calculated[-1] - self.history_VX_calculated[-k]) / dt
                dvy = (
                    self.history_VY_calculated[-1] - self.history_VY_calculated[-k]) / dt
            self.history_AX_calculated = np.append(
                self.history_AX_calculated, [dvx])
            self.history_AY_calculated = np.append(
                self.history_AY_calculated, [dvy])

    # , k_velocity=10, k_acceleration=2):
    def update(self, detection: Detection = None, mean=None):
        """Update the tracked object state with new detection.

        Parameters
        ----------
        detection : Detection, optional
            New Detection from yolo, by default None
        mean : List, optional
            Object values calculated by Kalman filter, by default None
        """
        if detection is not None:
            self.history.append(detection)
            self.history_X = np.append(self.history_X, [detection.X])
            self.history_Y = np.append(self.history_Y, [detection.Y])
            # self.update_velocity(k=k_velocity)
            # self.update_accel(k=k_acceleration)
            self.mean = mean
            self.X = detection.X
            self.Y = detection.Y
            VX_old = self.VX
            VY_old = self.VY
            self.VX = mean[4]
            self.VY = mean[5]
            self.history[-1].VX = self.VX
            self.history[-1].VY = self.VY
            # self.updateAccel(self.VX, VX_old, self.VY, VY_old)
            self.history[-1].AX = self.AX
            self.history[-1].AY = self.AY
            self.time_since_update = 0
        else:
            self.time_since_update += 1
        # it seems, that calculating euclidean distance of first and last stored detection, gives more accurate estimation of movement,
        # it can filter out phantom motion, when yolov7 detects a motionless object a bit off where it was last detected.
        # if mean is not None:
        #     if (abs(self.VX) < 1.0 and abs(self.VY) < 1.0):
        #         self.isMoving = False
        #     else:
        #         self.isMoving = True
        if (np.abs(self.VX) > 0.0 or np.abs(self.VY) > 0.0) and len(self.history) >= 5:
            # calculating euclidean distance of the first stored detection and last stored detection
            # this is still hard coded, so its a bit hacky, gotta find a good metric to tell if an object is moving or not
            self.isMoving = ((self.history[-5].X-self.history[-1].X)**2 + (
                self.history[-5].Y-self.history[-1].Y)**2)**(1/2) > 5.0
        else:
            self.isMoving = False
        # this is a fix for a specific problem, when an track is stuck, and showed as moving object
        # this is just a hack for now, TODO: find real solution
        # if len(self.history) == 2:
        #        self.bugged += 1
        # else:
        #    self.bugged = 0


def detectionFactory(objID, frameNum, label, confidence, x, y, width, height, vx, vy, ax, ay):
    """
    Create a Detection object.

    Parameters
    ----------
    objID : int
        Object ID, to which object the Detection belongs to.
    frameNum : int
        Frame number when the Detection occurred.
    label : str
        Label of the object, etc: car, person...
    confidence : float
        Confidence number, of how confident the neural network is in the detection.
    x : float
        X coordinate of the object.
    y : float
        Y coordinate of the object.
    width : float
        Width of the bounding box of the object.
    height : float
        Height of the bounding box of the object.
    vx : float
        Velocity on the X axis.
    vy : float
        Velocity on the Y axis.
    ax : float
        Acceleration on the X axis.
    ay : float
        Acceleration on the Y axis.

    Returns
    -------
    Detection
        The Detection object, which is to be returned.
    """
    retDet = Detection(label, confidence, x, y, width, height, frameNum)
    retDet.objID = objID
    retDet.VX = vx
    retDet.VY = vy
    retDet.AX = ax
    retDet.AY = ay
    return retDet


def trackedObjectFactory(detections: tuple):
    """
    Create a TrackedObject object from a list of detections.

    Parameters
    ----------
    detections : list
        A list of Detection objects.

    Returns
    -------
    TrackedObject
        The TrackedObject object created from the list of detections.
    """
    history, history_X, history_Y, history_VX_calculated, history_VY_calculated, history_AX_calculated, history_AY_calculated = detections
    tmpObj = TrackedObject(history[0].objID, history[0], len(detections))
    tmpObj.label = detections[0][-1].label
    tmpObj.history = history
    tmpObj.history_X = history_X
    tmpObj.history_Y = history_Y
    tmpObj.history_VX_calculated = history_VX_calculated
    tmpObj.history_VY_calculated = history_VY_calculated
    tmpObj.history_AX_calculated = history_AX_calculated
    tmpObj.history_AY_calculated = history_AY_calculated
    tmpObj.X = detections[0][-1].X
    tmpObj.Y = detections[0][-1].Y
    tmpObj.VX = detections[0][-1].VX
    tmpObj.VY = detections[0][-1].VY
    tmpObj.AX = detections[0][-1].AX
    tmpObj.AY = detections[0][-1].AY
    return tmpObj


def insert_weights_into_feature_vector(start: int, stop: int, n_weights: int, X: np.ndarray, Y: np.ndarray, insert_idx: int, feature_vector: np.ndarray) -> np.ndarray:
    """
    Insert coordinates into feature vector starting from the start_insert_idx index.

    Parameters
    ----------
    start : int
        First index of inserted coordinates.
    stop : int
        Stop index of coordinate vectors, which will not be inserted, this is the open end of the limits.
    n_weights : int
        Number of weights to be inserted.
    X : np.ndarray
        X coordinate array.
    Y : np.ndarray
        Y coordinate array.
    insert_idx : int
        The index where the coordinates will be inserted into the feature vector.
    feature_vector : np.ndarray
        The feature vector to insert the coordinates into.

    Returns
    -------
    np.ndarray
        The updated feature vector with the inserted coordinates.
    """
    retv = feature_vector.copy()
    stepsize = (stop-start)//n_weights
    assert n_weights, f"n_weights={n_weights} and max_stride are not compatible, lower n_weights or increase max_stride"
    weights_inserted = 0
    for widx in range(stop-1, start-1, -stepsize):
        if weights_inserted == n_weights:
            break
        retv = np.insert(retv, insert_idx, [X[widx], Y[widx]])
        weights_inserted += 1
    return retv


def findEnterAndExitPoints(path2db: str):
    """
    Extracts only the first and the last detections of tracked objects.

    Parameters
    ----------
    path2db : str
        Path to the database file.

    Returns
    -------
    tuple
        A tuple containing two lists: enterDetections and exitDetections.
        enterDetections : list
            List of first detections of objects.
        exitDetections : list
            List of last detections of objects.
    """
    rawDetectionData = databaseLoader.loadDetections(path2db)
    detections = detectionParser(rawDetectionData)
    rawObjectData = databaseLoader.loadObjects(path2db)
    trackedObjects = []
    for obj in tqdm.tqdm(rawObjectData, desc="Filter out enter and exit points."):
        tmpDets = []
        for det in detections:
            if det.objID == obj[0]:
                tmpDets.append(det)
        if len(tmpDets) > 0:
            trackedObjects.append(trackedObjectFactory(tmpDets))
    enterDetections = [obj.history[0] for obj in trackedObjects]
    exitDetections = [obj.history[-1] for obj in trackedObjects]
    return enterDetections, exitDetections


def detectionParser(rawDetectionData):
    """
    Convert raw detection data loaded from database to class Detection and numpy arrays.

    Parameters
    ----------
    rawDetectionData : list
        Raw values loaded from database.

    Returns
    -------
    tuple
        Tuple containing detections, and all the history numpy arrays.

    """
    detections = []
    history_X = np.array([])
    history_Y = np.array([])
    history_VX_calculated = np.array([])
    history_VY_calculated = np.array([])
    history_AX_calculated = np.array([])
    history_AY_calculated = np.array([])
    for entry in rawDetectionData:
        detections.append(detectionFactory(entry[0], entry[1], entry[2], entry[3], entry[4],
                          entry[5], entry[6], entry[7], entry[8], entry[9], entry[10], entry[11]))
        history_X = np.append(history_X, [entry[3]])
        history_Y = np.append(history_Y, [entry[4]])
        history_VX_calculated = np.append(history_VX_calculated, [entry[12]])
        history_VY_calculated = np.append(history_VY_calculated, [entry[13]])
        history_AX_calculated = np.append(history_AX_calculated, [entry[14]])
        history_AY_calculated = np.append(history_AY_calculated, [entry[15]])
    return (detections, history_X, history_Y, history_VX_calculated, history_VY_calculated, history_AX_calculated, history_AY_calculated)


def parseRawObject2TrackedObject(rawObjID: int, path2db: str):
    """
    Takes an objID and the path to the database, then returns a trackedObject object if detections can be assigned to the object.

    Parameters
    ----------
    rawObjID : int
        ID of an object.
    path2db : str
        Path to database.

    Returns
    -------
    trackedObject or bool
        TrackedObject object from dataManagement class, if no detections can be assigned to it, then returns False.

    """
    rawDets = databaseLoader.loadDetectionsOfObject(path2db, rawObjID)
    if len(rawDets) > 0:
        logging.debug(f"Detections loaded: {len(rawDets)} {rawDets[0]}")
        retTO = trackedObjectFactory(detectionParser(rawDets))
        return retTO
    else:
        return False


def preprocess_database_data(path2db: str):
    """
    Preprocesses database data (detections) by assigning detections to objects.

    Parameters
    ----------
    path2db : str
        Path to database file.

    Returns
    -------
    list
        List of object tracks.

    """
    rawObjectData = databaseLoader.loadObjects(path2db)
    trackedObjects = []
    for rawObj in tqdm.tqdm(rawObjectData, desc="Loading detections of tracks."):
        tmpDets = []
        rawDets = databaseLoader.loadDetectionsOfObject(path2db, rawObj[0])
        if len(rawDets) > 0:
            tmpDets = detectionParser(rawDets)
            trackedObjects.append(trackedObjectFactory(tmpDets))
    return trackedObjects


def preprocess_database_data_multiprocessed(path2db: str, n_jobs=None):
    """Preprocesses database data (detections) by assigning detections to objects.

    Parameters
    ----------
    path2db : str
        Path to database file.
    n_jobs : int, optional
        Number of parallel jobs to run, by default None.

    Returns
    -------
    List
        List of object tracks.
    """
    from multiprocessing import Pool
    rawObjectData = databaseLoader.loadObjects(path2db)
    tracks = []
    with Pool(processes=n_jobs) as pool:
        print("Preprocessing started.")
        start = time.time()
        results = pool.starmap_async(parseRawObject2TrackedObject, [
                                     [rawObj[0], path2db] for rawObj in rawObjectData])
        for result in tqdm.tqdm(results.get(), desc="Unpacking the result of detection assignment."):
            if result:
                tracks.append(result)
                logging.debug(f"{len(tracks)}")
        print(f"Detections assigned to Objects in {time.time()-start}s")
    return tracks


def tracks2joblib(path2db: str, n_jobs: int = 18):
    """Extract tracks from database and save them in a joblib object.

    Parameters
    ----------
    path2db : str
        Path to database.
    n_jobs : int, optional
        Number of parallel jobs to run, by default 18.

    Returns
    -------
    None
    """
    path = Path(path2db)
    tracks = preprocess_database_data_multiprocessed(path2db, n_jobs)
    savepath = path.with_suffix(".joblib")
    print('Saving: ', savepath)
    joblib.dump(tracks, savepath)
