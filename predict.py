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
import cv2 as cv

def movementIsRight(X_hist, historyDepth):
    """Returns True if object moving left to right or False if right to left

    Args:
        X_hist (list(int)): list of X coordinates of length historyDepth
        historyDepth (int): the length of the x coord list

    Returns:
        bool: retval
    """
    try:
        return (X_hist[-1] > X_hist[-historyDepth])
    except:
        return (X_hist[-1] > X_hist[0])

def predictLinear(trackedObject, linear_model, historyDepth=3, futureDepth=30, image=None):
    """Calculating future trajectory of the trackedObject

    Args:
        trackedObject (TrackedObject): the tracked Object
        linear_model (sklearn linear_model): Linear model used to calculate the trajectory
        historyDepth (int, optional): the number of detections that the trajectory should be calculated from. Defaults to 3.
        futureDepth (int, optional): how far in the future should we predict. Defaults to 30.
        image (Opencv image, optional): if image is inputted, then trajectories are drawn to the image. Defaults to None.

    """
    X_train = np.array([det.X for det in trackedObject.history[-historyDepth-1:-1]])
    y_train = np.array([det.Y for det in trackedObject.history[-historyDepth-1:-1]])
    if len(X_train) >= 3 and len(y_train) >= 3:
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        reg = linear_model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred
        if image is not None:
            for x,y in zip(trackedObject.futureX, trackedObject.futureY):
                cv.circle(image, (int(x),int(y)), 1, color=(0,0,255))

def predictPoly():
    # TODO: implement
    pass

def draw_predictions(trackedObject, image, frameNumber):
    """Draw prediction information to image

    Args:
        trackedObject (TrackedObject): tracked object
        image (Opencv Image): image to draw on
        frameNumber (int): current number of the video frame
    """
    if trackedObject.history[-1].frameID == frameNumber:
        idx = 0
        for x, y in zip(trackedObject.futureX, trackedObject.futureY):
            if (idx % 4) == 0:
                cv.circle(image, (int(x), int(y)), 1, color=(0,0,255))
            idx += 1