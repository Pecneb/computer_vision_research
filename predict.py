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
from email.mime import base
from random import random
from statistics import mode
from cv2 import trace
import numpy as np
import cv2 as cv
from sklearn import linear_model, kernel_ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import KernelCenterer, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

from historyClass import TrackedObject

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

def predictLinear(trackedObject, historyDepth=3, futureDepth=30, image=None):
    """Calculating future trajectory of the trackedObject

    Args:
        trackedObject (TrackedObject): the tracked Object
        linear_model (sklearn linear_model): Linear model used to calculate the trajectory
        historyDepth (int, optional): the number of detections that the trajectory should be calculated from. Defaults to 3.
        futureDepth (int, optional): how far in the future should we predict. Defaults to 30.
        image (Opencv image, optional): if image is inputted, then trajectories are drawn to the image. Defaults to None.

    """
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # k (int) : number of training points
        k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in x_history[-historyDepth:-1:slice]])
        y_train = np.array([y for y in y_history[-historyDepth:-1:slice]])
        # check if the movement is right or left, becouse the generated x_test vector
        # if movement is right vector is ascending, otherwise descending
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # fit linear model on the x_train vectors points
        model = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(), random_state=30, min_samples=X_train.reshape(-1,1).shape[1]+1)
        reg = model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred

def predictPoly(trackedObject, degree=3, historyDepth=3, futureDepth=30):
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # k (int) : number of training points
        k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in x_history[-historyDepth:-1:slice]])
        y_train = np.array([y for y in y_history[-historyDepth:-1:slice]])
        # generating future points
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # poly features
        polyModel = make_pipeline(PolynomialFeatures(degree), linear_model.RANSACRegressor(base_estimator=linear_model.Ridge(alpha=0.5), random_state=30, min_samples=X_train.reshape(-1,1).shape[1]+1))
        polyModel.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
        y_pred = polyModel.predict(X_test.reshape(-1, 1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred

def predictSpline(trackedObject, degree=3, historyDepth=3, futureDepth=30):
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # calculating even slices to pick k points to fit linear model on
        X_train = np.array([x for x in x_history[-historyDepth:-1]])
        y_train = np.array([y for y in y_history[-historyDepth:-1]])
        # generating future points
        if movementIsRight(X_train, historyDepth):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # poly features
        polyModel = make_pipeline(SplineTransformer(), linear_model.Ridge(alpha=1e-3)) 
        polyModel.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
        y_pred = polyModel.predict(X_test.reshape(-1, 1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred


def predictMixed(trackedObject, historyDepth=3, futureDepth=30):
    predictPoly(trackedObject, historyDepth=historyDepth, futureDepth=futureDepth)
    for idx in range(1, len(trackedObject.futureY)-1):
        if ((trackedObject.futureY[idx] > trackedObject.futureY[idx-1] and \
            trackedObject.futureY[idx] > trackedObject.futureY[idx+1]) \
            or (trackedObject.futureY[idx] < trackedObject.futureY[idx-1] and \
            trackedObject.futureY[idx] < trackedObject.futureY[idx+1]) or trackedObject.futureY[idx] < 0):
            predictLinear(trackedObject, futureDepth=futureDepth, historyDepth=historyDepth)
            break

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

def draw_history(trackedObject, image, frameNumber):
    if trackedObject.history[-1].frameID == frameNumber:
        idx = 0
        for det in trackedObject.history:
            if (idx % 1) == 0:
                cv.circle(image, (int(det.X), int(det.Y)), 1, color=(0,255,0))
            idx += 1