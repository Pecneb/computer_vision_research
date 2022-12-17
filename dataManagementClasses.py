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

from dataclasses import dataclass, field
import numpy as np

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
    """
    label: str
    confidence: float
    X: int
    Y: int
    Width: int
    Height: int
    frameID: int
    # These variables are optional, not needed in the main loop. These are used in the analyzer script and databaseLoader script.
    VX: int = field(init=False,)
    VY: int = field(init=False)
    AX: int = field(init=False)
    AY: int = field(init=False)
    objID: int = field(init=False)

    def __repr__(self) -> str:
        return f"Label: {self.label}, Confidence: {self.confidence}, X: {self.X}, Y: {self.Y}, Width: {self.Width}, Height: {self.Height}, Framenumber: {self.frameID}"

@dataclass
class TrackedObject():
    """Class for storing a detected object's tracking history
    
       Fields:
        objID(int): a unique identification number of the detected object
        label(str): the label of detected object ex. car, person, etc.
        history(list of Detections): list of Detection that are propably the same object 
        futureX(int): the predicrted X position
        futureY(int): the predicted Y position
        isMoving(bool): True if object is in motion
        time_since_update(int): keeping track of missed detections of the object
        max_age(int): when time_since_update hits max_age, tracking is deleted
        mean(list[int]): output of kalman filter, (x,y,a,h,vx,vy,va,vh)
        X(int): last detected x coord of the object
        Y(int): last detected y coord of the object
        VX(float): last calculated velocity of X
        VY(float): last calculated velocity of Y
        bugged(int): int value to count stuck tracks, this happens when the program is running for a long time, this is a bit hacky but necessary, still dont know what causes this problem
        Methods:
         avgArea(): returns the average bbox area of all the detections in the history
         update(): called when tracking can be updated
    """
    objID: int
    label: int = field(init=False)
    futureX: list = field(init=False)
    futureY: list = field(init=False)
    history: list
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
    featureVector: np.ndarray = field(init=False)

    def __init__(self, id, first, max_age=30):
        self.objID = id
        self.history = [first]
        self.X = first.X
        self.Y = first.Y
        self.VX = 0 
        self.VY = 0
        self.AX = 0
        self.AY = 0
        self.label = first.label
        self.isMoving = False
        self.futureX = []
        self.futureY = []
        self.max_age = max_age
        self.time_since_update = 0
        self.mean = []
        self.featureVector = np.array() # feature vector that describes the track 
        #self.bugged = 0 
    
    def __repr__(self) -> str:
        return "Label: {}, ID: {}, X: {}, Y: {}, VX: {}, VY: {}, Age: {}, ActualHistoryLength: {}".format(self.label, self.objID, self.X, self.Y, self.VX, self.VY, self.time_since_update, len(self.history))

    def avgArea(self):
        areas = [(det.Width*det.Height) for det in self.history]
        return np.average(areas)
    
    def updateAccel(self, new_vx, old_vx, new_vy, old_vy):
        """Calculate accelaration based on kalman filters velocity.
        Normal accelaration calculation (new_v - old_v) / (new_time - old_time).
        The time between the two detection is the (new_time-old_time).

        Args:
            new_vx (float): New velocity of x 
            old_vx (float): Old velocity of x from previous detection
            new_vy (float): New velocity of y
            old_vy (float): Old velocity of y from previous detection
        """
        self.AX = (new_vx - old_vx) / (self.time_since_update+1)
        self.AY = (new_vy - old_vy) / (self.time_since_update+1)

    def upscale_feature(featureVector: np.ndarray, framewidth: int, frameheight: int):
        """Rescale normalized coordinates with the given frame sizes.

        Args:
            featureVector (np.ndarray): Feature vector of the track.
            framewidth (int): Width of the video frame. 
            frameheight (int): Height of the video frame. 

        Returns:
            np.ndarray: upscaled feature vector
        """
        return np.array([framewidth*featureVector[0], frameheight*featureVector[1], framewidth*featureVector[2],
                        frameheight*featureVector[3], framewidth*featureVector[4], frameheight*featureVector[5],
                        framewidth*featureVector[6], frameheight*featureVector[7], framewidth*featureVector[8],
                        frameheight*featureVector[9]])

    def downscale_feature(featureVector: np.ndarray, framewidth: int, frameheight: int):
        """Downscale coordinates to normalized coordinates with the given frame sizes.

        Args:
            featureVector (np.ndarray): Feature vector of the track.
            framewidth (int): Width of the video frame. 
            frameheight (int): Height of the video frame. 

        Returns:
            np.ndarray: downscaled feature vector
        """
        return np.array([featureVector[0] / framewidth, featureVector[1] / frameheight, featureVector[2] / framewidth,
                        featureVector[3] / frameheight, featureVector[4] / framewidth, featureVector[5] / frameheight,
                        featureVector[6] / framewidth, featureVector[7] / frameheight, featureVector[8] / framewidth,
                        featureVector[9] / frameheight])
                        
    def update_feature(self, framewidth: int, frameheight: int):
        """Update the track's feature vector.

        Args:
            framewidth (int): Width of the video frame.
            frameheight (int): Height of the video frame. 
        """
        n = len(self.history)
        self.featureVector = np.array([self.history[0].X, self.history[0].Y, self.history[0].VX, 
                                    self.history[0].VY, self.history[n//2].X, self.history[n//2].Y, 
                                    self.history[n].X, self.history[n].Y, self.history[n].VX, 
                                    self.history[n].VY])

    def update(self, detection=None, mean=None, historyDepth = 30):
        """Update tracking

        Args:
            detection (Detection, optional): historyClass Detecton object. If none, increment time_since_update. Defaults to None.
            features (list[int], optional): x, y, a, h, vx, vy, va, h --> coordinates, aspect ratio, height and their velocities. Defaults to None.
        """
        if detection is not None and mean is not None:
            self.history.append(detection)
            self.mean = mean 
            self.X = mean[0]
            self.Y = mean[1]
            VX_old = self.VX
            VY_old = self.VY
            self.VX = mean[4]
            self.VY = mean[5]
            self.updateAccel(self.VX, VX_old, self.VY, VY_old)
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
        if len(self.history) > historyDepth:
            # calculating euclidean distance of the first stored detection and last stored detection
            # this is still hard coded, so its a bit hacky, gotta find a good metric to tell if an object is moving or not
            self.isMoving = ((self.history[0].X-self.history[-1].X)**2 + (self.history[0].Y-self.history[-1].Y)**2)**(1/2) > 7.0  
        if not self.isMoving:
            self.VX = 0.0
            self.VY = 0.0
        # this is a fix for a specific problem, when an track is stuck, and showed as moving object
        # this is just a hack for now, TODO: find real solution
        #if len(self.history) == 2:
        #        self.bugged += 1
        #else:
        #    self.bugged = 0