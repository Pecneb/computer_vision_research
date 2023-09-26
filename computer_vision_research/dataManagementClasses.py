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

import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Detection:
    """Class for storing detections of YOLO Darknet
        
       Fields: 
        label(string): classification name ex. car, person, bicycle, etc.
        confidence(float): the likeliness that the detected object is really the above label, 0.0-1.0
        X(int): center of bounding box in the X axis
        Y(int): center of bounding box in the Y axis
        Width(int): width of the bounding box
        Height(int): height of the bounding box
        frameID(int): the number of the frame, the detection is oqqured
        VX(int)
    """
    label: str
    confidence: float
    X: float 
    Y: float 
    Width: float 
    Height: float 
    frameID: int
    # These variables are optional, not needed in the main loop. These are used in the analyzer script and databaseLoader script.
    VX: float = field(init=False,)
    VY: float = field(init=False)
    AX: float = field(init=False)
    AY: float = field(init=False)
    objID: int = field(init=False)

    def __repr__(self) -> str:
        return f"Label: {self.label}, Confidence: {self.confidence}, X: {self.X}, Y: {self.Y}, Width: {self.Width}, Height: {self.Height}, Framenumber: {self.frameID}"
    
    def __eq__(self, other) -> bool:
        if self.label != other.label: return False
        if self.confidence != other.confidence : return False 
        if self.X != other.X: return False 
        if self.Y != other.Y: return False 
        if self.Width != other.Width: return False 
        if self.Height != other.Height: return False 
        if self.frameID != other.frameID: return False 
        return True

@dataclass
class TrackedObject():
    """This class is used to represent a tracked objects lifetime.
    One can say a tracked object's trajectory.


    Attributes
    ----------
    objID : int
        unique identification number of the object
    label : str
        class label of the object ie. car, person, etc.
    futureX : list 
        predicted X coordinates, this attribute is not used anymore, it is being held, due to compatibility reasons
    futureY : list 
        predicted Y coordinates, this attribute is not used anymore, it is being held, due to compatibility reasons
    history : ndarray
        a list of previous detections of the object        
    history_X : ndarray
        numpy array of previous X coordintates
    history_Y : ndarray
        numpy array of previous Y coordintates
    history_VX_calculated : ndarray
        the velocity of dimension X calculated from the history_X numpy array
    history_VY_calculated : ndarray
        the velocity of dimension Y calculated from the history_Y numpy array
    history_AX_calculated : ndarray
        the acceleration of dimension X calculated from the history_VX_calculated numpy array
    history_AY_calculated : ndarray
        the acceleration of dimension Y calculated from the history_VY_calculated numpy array
    isMoving: bool
        a boolean value, that tells if the object is moving or not
    time_since_update : int
        the number of frames since the last detection of the object
    max_age : int
        the maximum number of frames, that an object can be tracked without a detection,
        if the time_since_update attribute reaches this value, the object is deleted from the tracker
    mean : list
        a list of the mean values of the detections, this is used for the kalman filter
    X : int
        the X coordinate of the last detection
    Y : int
        the Y coordinate of the last detection
    VX : float
        the velocity of dimension X calculated from the history_X numpy array
    VY : float
        the velocity of dimension Y calculated from the history_Y numpy array
    AX : float
        the acceleration of dimension X calculated from the history_VX_calculated numpy array
    AY : float
        the acceleration of dimension Y calculated from the history_VY_calculated numpy array

    Returns:
        _type_: _description_
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
    time_since_update : int = field(init=False)
    max_age : int
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

<<<<<<< HEAD:dataManagementClasses.py
    def __init__(self, id, first, max_age=30):
=======
    def __init__(self, id: int, first: Detection, max_age: int =30):
>>>>>>> gui_app:computer_vision_research/dataManagementClasses.py
        """Constructor method for TrackedObject class.

        Args:
            id (int): Unique identification number of the object. 
            first (Detection): First detection of the object.
            max_age (int, optional): The maximum frame number an object
                can be tracked. Defaults to 30.
        """
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
        #self.bugged = 0 
        self._dataset = ""
    
    def __repr__(self) -> str:
        return "Label: {}, ID: {}, X: {:10.4f}, Y: {:10.4f}, VX: {:10.4f}, VY: {:10.4f}, Age: {}, ActualHistoryLength: {}".format(self.label, self.objID, self.X, self.Y, self.history_VX_calculated[-1], self.history_VY_calculated[-1], self.time_since_update, len(self.history))
    
    def __hash__(self) -> int:
        retval = int(self.objID+np.sum([self.history[i].frameID for i in range(len(self.history))]))
        # print(retval, self.objID, self._dataset)
        return retval
    
    def __eq__(self, other) -> bool:
        for i in range(len(self.history)):
            for j in range(len(other.history)):
                if self.history[i] != other.history[j]:
                    return False
        return self.objID == other.objID

    def avgArea(self):
        """Calculate average area of the bounding boxes of the object.

        Returns:
            float: Average area of the bounding boxes of the object. 
        """
        areas = [(det.Width*det.Height) for det in self.history]
        return np.average(areas)
    
    def updateAccel(self, new_vx, old_vx, new_vy, old_vy):
        """Calculate acceleration based on kalman filters velocity.
        Normal acceleration calculation (new_v - old_v) / (new_time - old_time).
        The time between the two detection is the (new_time-old_time).

        Args:
            new_vx (float): New velocity of x 
            old_vx (float): Old velocity of x from previous detection
            new_vy (float): New velocity of y
            old_vy (float): Old velocity of y from previous detection
        """
        self.AX = (new_vx - old_vx) / (self.time_since_update+1)
        self.AY = (new_vy - old_vy) / (self.time_since_update+1)

    @staticmethod
    def upscale_feature(featureVector: np.ndarray, framewidth: int = 1920, frameheight: int = 1080):
        """Rescale normalized coordinates with the given frame sizes.

        Args:
            featureVector (np.ndarray): Feature vector of the track.
            framewidth/ratio (int): Width of the video frame. 
            frameheight (int): Height of the video frame. 

        Returns:
            np.ndarray: upscaled feature vector
        """
        if featureVector is None:
            return None
        ratio = framewidth / frameheight

        ret_featureVector = np.array([])
        for i in range(featureVector.shape[0]):
            if i == 0 or i % 2 == 0:
                ret_featureVector = np.append(ret_featureVector, [featureVector[i]*framewidth/ratio])
            else:
                ret_featureVector = np.append(ret_featureVector, [frameheight*featureVector[i]])
        return ret_featureVector

        """
        if feature_v3:
            return np.array([framewidth/ratio*featureVector[0], frameheight*featureVector[1], 
                            framewidth/ratio*featureVector[2], frameheight*featureVector[3], 
                            framewidth/ratio*featureVector[4], frameheight*featureVector[5]])
        return np.array([framewidth/ratio*featureVector[0], frameheight*featureVector[1], framewidth/ratio*featureVector[2],
                        frameheight*featureVector[3], framewidth/ratio*featureVector[4], frameheight*featureVector[5],
                        framewidth/ratio*featureVector[6], frameheight*featureVector[7], framewidth/ratio*featureVector[8],
                        frameheight*featureVector[9]])

        """

    @staticmethod
    def downscale_feature(featureVector: np.ndarray, framewidth: int = 1920, frameheight: int = 1080):
        """Rescale normalized coordinates with the given frame sizes.

        Args:
            featureVector (np.ndarray): Feature vector of the track.
            framewidth/ratio (int): Width of the video frame. 
            frameheight (int): Height of the video frame. 

        Returns:
            np.ndarray: upscaled feature vector
        """
        if featureVector is None:
            return None
        ratio = framewidth / frameheight

        ret_featureVector = np.array([])
        for i in range(featureVector.shape[0]):
            if i == 0 or i % 2 == 0:
                ret_featureVector = np.append(ret_featureVector, [featureVector[i]/framewidth*ratio])
            else:
                ret_featureVector = np.append(ret_featureVector, [featureVector[i]/frameheight])
        return ret_featureVector

        """
        if feature_v3:
            return np.array([featureVector[0] / framewidth*ratio, featureVector[1] / frameheight, 
                            featureVector[2] / framewidth*ratio, featureVector[3] / frameheight, 
                            featureVector[4] / framewidth*ratio, featureVector[5] / frameheight,])
        return np.array([featureVector[0] / framewidth*ratio, featureVector[1] / frameheight, featureVector[2] / framewidth*ratio,
                        featureVector[3] / frameheight, featureVector[4] / framewidth*ratio, featureVector[5] / frameheight,
                        featureVector[6] / framewidth*ratio, featureVector[7] / frameheight, featureVector[8] / framewidth*ratio,
                        featureVector[9] / frameheight])

        """
                        
    def feature_(self):
        """Return feature vector of track.
        A simple feature vector consisting,
        of the first, middle and last detection's X and Y coordinates.
        The first and last detection's velocity is also included.
        """
        n = len(self.history)-1
        if n < 3:
            return None
        return np.array([self.history[0].X, self.history[0].Y, self.history[0].VX, 
                        self.history[0].VY, self.history[n//2].X, self.history[n//2].Y, 
                        self.history[n].X, self.history[n].Y, self.history[n].VX, 
                        self.history[n].VY])
    
    def feature_v3_v4(self):
        """Return feature vector of track.
        A simple feature vector consisting,
        if the first, middle and last detection's X and Y coordinates.
        """
        n = len(self.history)-1
        if n < 3:
            return None
        return np.array([self.history[0].X, self.history[0].Y,
                        self.history[n//2].X, self.history[n//2].Y,
                        self.history[n].X, self.history[n].Y])

    """This is the same as the v3 feature vector.
    def feature_v4(self):
        n = len(self.history)-1
        if n < 3:
            return None
        return np.array([self.history[0].X, self.history[0].Y, 
                        self.history[n//2].X, self.history[n//2].Y, 
                        self.history[n].X, self.history[n].Y])
    """
    
    def feature_v7(self):
        """Return version 7 feature vector.
        The first and last detection's X and Y coordinates.
        Also the velocities of the first and last detection.
        The feature vector is weighted by a hardcoded value.
        [frist.X, first.Y, first.VX, first.VY, last.X, last.Y, last.VX, last.VY] * [1, 1, 100, 100, 2, 2, 200, 200]
        """
        n = len(self.history_X)-1
        if self.history_X.shape[0] < self.max_age:
            return None
        return np.array([self.history_X[0], self.history_Y[0], 
                        self.history_VX_calculated[0], self.history_VY_calculated[0], 
                        self.history_X[n], self.history_Y[n],
                        self.history_VX_calculated[n-1], self.history_VY_calculated[n-1]]) * np.array([1, 1, 100, 100, 2, 2, 200, 200])

    def feature_v5(self, n_weights: int):
        """Return version 5 feature vector.
        This feature vector consists, of
        the first and last detection's X and Y coordinates,
        Also the n_weight number controls the number of
        inserted coordinates between the first and the last
        detection's coordinates.

        Args:
            n_weights (int):  

        Returns:
            feature: A numpy ndarray. 
        """
        n = (self.history_X.shape[0] // 2)
        if (n // n_weights) < 1:
            return None
        feature = np.array([self.history_X[0], self.history_Y[0],
                            self.history_X[-1], self.history_Y[-1]])
        feature = insert_weights_into_feature_vector(n, self.history_X.shape[0], n_weights, self.history_X, self.history_Y, 2, feature)
        return feature


    def update_velocity(self, k: int = 10):
        """Calculate velocity from X,Y coordinates.
        """
        if self.history_X.shape[0] < k:
            self.history_VX_calculated = np.append(self.history_VX_calculated, [0]) 
            self.history_VY_calculated = np.append(self.history_VY_calculated, [0]) 
        else:
            #self.history_DT = np.append(self.history_DT, [self.history[-1].frameID - self.history[-k].frameID])
            #if self.history_DT[-1] == 0:
            #    dx = 0 
            #    dy = 0
            #else:
            dx = (self.history_X[-1] - self.history_X[-k]) / (self.history[-1].frameID - self.history[-k].frameID)
            dy = (self.history_Y[-1] - self.history_Y[-k]) / (self.history[-1].frameID - self.history[-k].frameID)
            self.history_VX_calculated = np.append(self.history_VX_calculated, [dx]) 
            self.history_VY_calculated = np.append(self.history_VY_calculated, [dy])  
        self.history_VT = np.append(self.history_VT, [self.history[-1].frameID])


    def update_accel(self, k: int = 2):
        """Calculate velocity from X,Y coordinates.
        """
        if self.history_VX_calculated.shape[0] < k:
            self.history_AX_calculated = np.append(self.history_AX_calculated, [0]) 
            self.history_AY_calculated = np.append(self.history_AY_calculated, [0]) 
        else:
            dt =(self.history_VT[-1] - self.history_VT[-k]) 
            if dt == 0:
                dvx = 0 
                dvy = 0
            else:
                dvx = (self.history_VX_calculated[-1] - self.history_VX_calculated[-k]) / dt 
                dvy = (self.history_VY_calculated[-1] - self.history_VY_calculated[-k]) / dt
            self.history_AX_calculated = np.append(self.history_AX_calculated, [dvx]) 
            self.history_AY_calculated = np.append(self.history_AY_calculated, [dvy])  

    def update(self, detection=None, mean=None, k_velocity=10, k_acceleration=2, historyDepth = 30):
        """Update tracking.

        Args:
            detection (Detection, optional): historyClass Detecton object. If none, increment time_since_update. Defaults to None.
            features (list[int], optional): x, y, a, h, vx, vy, va, h --> coordinates, aspect ratio, height and their velocities. Defaults to None.
        """
        if detection is not None and mean is not None:
            self.history.append(detection)
            self.history_X = np.append(self.history_X, [detection.X])
            self.history_Y = np.append(self.history_Y, [detection.Y])
            self.update_velocity(k=k_velocity)
            self.update_accel(k=k_acceleration)
            self.mean = mean 
            self.X = detection.X 
            self.Y = detection.Y
            VX_old = self.VX
            VY_old = self.VY
            self.VX = mean[4]
            self.VY = mean[5]
            self.history[-1].VX = self.VX
            self.history[-1].VY = self.VY
            self.updateAccel(self.VX, VX_old, self.VY, VY_old)
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
        if (np.abs(self.history_VX_calculated[-1]) > 0.0 or np.abs(self.history_VY_calculated[-1]) > 0.0) and len(self.history) >= 5:
            # calculating euclidean distance of the first stored detection and last stored detection
            # this is still hard coded, so its a bit hacky, gotta find a good metric to tell if an object is moving or not
            self.isMoving = ((self.history[-5].X-self.history[-1].X)**2 + (self.history[-5].Y-self.history[-1].Y)**2)**(1/2) > 5.0  
        else:
            self.isMoving = False
        # this is a fix for a specific problem, when an track is stuck, and showed as moving object
        # this is just a hack for now, TODO: find real solution
        #if len(self.history) == 2:
        #        self.bugged += 1
        #else:
        #    self.bugged = 0

def detectionFactory(objID: int, frameNum: int, label: str, confidence: float, x: float, y: float, width: float, height: float, vx: float, vy: float, ax:float, ay: float):
    """Create Detection object.

    Args:
        objID (int): Object ID, to which object the Detection belongs to. 
        frameNum (int): Frame number when the Detection occured. 
        label (str): Label of the object, etc: car, person... 
        confidence (float): Confidence number, of how confident the neural network is in the detection. 
        x (float):  X coordinate of the object.
        y (float): Y coordinate of the object. 
        width (float): Width of the bounding box of the object. 
        height (float): Height of the bounging box of the object. 
        vx (float): Velocity on the X axis. 
        vy (float): Velocity on the Y axis. 
        ax (float): Acceleration on the X axis. 
        ay (float): Acceleration on the Y axis. 

    Returns:
        Detection: The Detection object, which is to be returned. 
    """
    retDet = Detection(label, confidence, x,y,width,height,frameNum)
    retDet.objID = objID
    retDet.VX = vx
    retDet.VY = vy
    retDet.AX = ax
    retDet.AY = ay
    return retDet

def trackedObjectFactory(detections: tuple):
    """Create trackedObject object from list of detections

    Args:
        detections (list): list of detection 

    Returns:
        TrackedObject:  trackedObject
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

def insert_weights_into_feature_vector(start: int, stop: int, n_weights: int, X: np.ndarray, Y: np.ndarray, insert_idx: int, feature_vector: np.ndarray):
    """Insert coordinates into feature vector starting from the start_insert_idx index.

    Args:
        start (int): first index of inserted coordinates 
        stop (int): stop index of coordinate vectors, which will not be inserted, this is the open end of the limits
        n_weights (int): number of weights to be inserted
        X (ndarray): x coordinate array
        Y (ndarray): y coordinate array
        start_insert_idx (int): the index where the coordinates will be inserted into the feature vector 
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