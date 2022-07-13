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
from numpy import average

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
    cls: int
    confidence: float
    X: int
    Y: int
    Width: int
    Height: int
    frameID: int

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
        Methods:
         avgArea(): returns the average bbox area of all the detections in the history
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

    def __init__(self, id, first, max_age=30):
        self.objID = id
        self.history = [first]
        self.X = first.X
        self.Y = first.Y
        self.VX = 0 
        self.VY = 0
        self.label = first.label
        self.isMoving = False
        self.futureX = []
        self.futureY = []
        self.max_age = max_age
        self.time_since_update = 0
        self.mean = [] 
    
    def __repr__(self) -> str:
        return "ID: {}, Label: {}, Moving: {}, Age: {}, X: {}, Y: {}, VX: {}, VY: {}".format(self.objID, self.label, self.isMoving, self.time_since_update, self.X, self.Y, self.VX, self.VY)

    def avgArea(self):
        areas = [(det.Width*det.Height) for det in self.history]
        return average(areas)

    def update(self, detection=None, mean=None):
        """Update tracking

        Args:
            detection (Detection, optional): historyClass Detecton object. If none, increment time_since_update. Defaults to None.
            features (list[int], optional): x, y, a, h, vx, vy, va, h --> coordinates, aspect ratio, height and their velocities. Defaults to None.
        """
        if detection is not None:
            self.history.append(detection)
            self.mean = mean 
            self.X = mean[0]
            self.Y = mean[1]
            self.VX = mean[4]
            self.VY = mean[5]
            self.time_since_update = 0 
        else:
            self.time_since_update += 1
        # it seem, that calculating euclidean distance of first and last stored detection, gives more accurate estimation of movement
        # if mean is not None:
        #     if (abs(self.VX) < 1.0 and abs(self.VY) < 1.0):
        #         self.isMoving = False
        #     else:
        #         self.isMoving = True
        if self.history:
            # calculating euclidean distance of the first stored detection and last stored detection
            # this is still hard coded, so its a bit hacky, gotta find a good metric to tell if an object is moving or not
            self.isMoving = ((self.history[0].X-self.history[-1].X)**2 + (self.history[0].Y-self.history[-1].Y)**2)**(1/2) > 7.0 