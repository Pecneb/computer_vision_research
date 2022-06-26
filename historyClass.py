from dataclasses import dataclass, field

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
        history(list of Detections): list of Detection that are propably the same object 
        futureX(int): the predicrted X position
        futureY(int): the predicted Y position
    """
    objID: int
    history: list[Detection] = field(default_factory=list, init=False) 
    futureX: int
    futureY: int