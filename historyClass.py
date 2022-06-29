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
        Methods:
         avgArea(): returns the average bbox area of all the detections in the history
    """
    objID: int
    label: int = field(init=False)
    futureX: list[int] = field(init=False)
    futureY: list[int] = field(init=False)
    history: list[Detection]
    isMoving: bool = field(init=False)

    def __init__(self, id, first):
        self.objID = id
        self.history = [first]
        self.label = first.label
        self.isMoving = False

    def avgArea(self):
        areas = [(det.Width*det.Height) for det in self.history]
        return average(areas)