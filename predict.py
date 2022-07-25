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
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

from historyClass import TrackedObject

def movementIsRight(obj: TrackedObject):
    """Returns true, if the object moving right, false otherwise. 

    Args:
        obj (TrackedObject): tracking data of an object 
    
    Return:
        bool: Tru if obj moving right.

    """
    return obj.VX > 0 
    
def predictLinear(trackedObject: TrackedObject, k=3, historyDepth=3, futureDepth=30):
    """Fit linear function on detection history of an object, to predict future coordinates.

    Args:
        trackedObject (TrackedObject): The object, which's future coordinates should be predicted. 
        k (int, optional): Number of training points, ex.: if historyDepth is 30 and k is 3, then the 1st, 15th and 30th points will be training points. Defaults to 3.
        historyDepth (int, optional): Training history length. Defaults to 3.
        futureDepth (int, optional): Prediction vectors length. Defaults to 30.
    """
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # k (int) : number of training points
        # k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in x_history[-historyDepth:-1:slice]])
        y_train = np.array([y for y in y_history[-historyDepth:-1:slice]])
        # check if the movement is right or left, becouse the generated x_test vector
        # if movement is right vector is ascending, otherwise descending
        if movementIsRight(trackedObject):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # fit linear model on the x_train vectors points
        model = linear_model.LinearRegression(n_jobs=-1)
        reg = model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred

def predictPoly(trackedObject: TrackedObject, degree=3, k=3, historyDepth=3, futureDepth=30):
    """Fit polynomial function on detection history of an object, to predict future coordinates.

    Args:
        trackedObject (TrackedObject): The object, which's future coordinates should be predicted. 
        degree (int, optional): The polynomial functions degree. Defaults to 3.
        k (int, optional): Number of training points, ex.: if historyDepth is 30 and k is 3, then the 1st, 15th and 30th points will be training points. Defaults to 3.
        historyDepth (int, optional): Training history length. Defaults to 3.
        futureDepth (int, optional): Prediction vectors length. Defaults to 30.
    """
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # k (int) : number of training points
        # k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in x_history[-historyDepth:-1:slice]])
        y_train = np.array([y for y in y_history[-historyDepth:-1:slice]])
        # generating future points
        if movementIsRight(trackedObject):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # poly features
        polyModel = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge(alpha=1e-3))
        polyModel.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
        # print(X_train.shape, y_train.shape)
        y_pred = polyModel.predict(X_test.reshape(-1, 1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred

def predictSpline(trackedObject: TrackedObject, k=3, degree=3, n_knots=4, historyDepth=3, futureDepth=30):
    """TODO: Spline implementation.
    This function is not working properly yet. Usage is not advised.

    Args:
        trackedObject (TrackedObject): _description_
        degree (int, optional): _description_. Defaults to 3.
        historyDepth (int, optional): _description_. Defaults to 3.
        futureDepth (int, optional): _description_. Defaults to 30.
    """
    # TODO: still not working, got to experiment with it more
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 4 and len(y_history) >= 4:
        l = len(x_history) // 2 * 2
        X_train = np.zeros((l)) 
        y_train = np.zeros((l))
        for i in range(0, l):
            X_train[i] = x_history[i]
            y_train[i] = y_history[i]
        # generating future points
        if movementIsRight(trackedObject):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth, num=futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth, num=futureDepth)
        # poly features
        splineModel= make_pipeline(SplineTransformer(degree=degree, n_knots=n_knots), linear_model.Ridge(alpha=1e-3)) 
        splineModel.fit(X_train[:, np.newaxis], y_train)
        y_pred = np.array(splineModel.predict(X_test[:, np.newaxis]))
        
        trackedObject.futureX = X_test.reshape(futureDepth)
        trackedObject.futureY = y_pred
        


def predictLinPoly(trackedObject: TrackedObject, k=3, historyDepth=3, futureDepth=30):
    """Fit linear or polynomial function on history data based on prediction results. 
    This function intended to filter out extreme polynomial function results.
    
    Args:
        trackedObject (TrackedObject): The object, which's future coordinates should be predicted. 
        degree (int, optional): The polynomial functions degree. Defaults to 3.
        k (int, optional): Number of training points, ex.: if historyDepth is 30 and k is 3, then the 1st, 15th and 30th points will be training points. Defaults to 3.
        historyDepth (int, optional): Training history length. Defaults to 3.
        futureDepth (int, optional): Prediction vectors length. Defaults to 30.
    """
    predictPoly(trackedObject, degree=2, historyDepth=historyDepth, futureDepth=futureDepth)
    for idx in range(1, len(trackedObject.futureY)-1):
        if ((trackedObject.futureY[idx] > trackedObject.futureY[idx-1] and trackedObject.futureY[idx] > trackedObject.futureY[idx+1]) or \
            (trackedObject.futureY[idx] < trackedObject.futureY[idx-1] and trackedObject.futureY[idx] < trackedObject.futureY[idx+1]) or \
            trackedObject.futureY[idx] < 0):
            predictLinear(trackedObject, futureDepth=futureDepth, historyDepth=historyDepth)
            break

def predictLinSpline(trackedObject: TrackedObject, historyDepth=3, futureDepth=30):
    """TODO: Spline implementation.
    This function is not working properly yet. Usage is not advised.

    Args:
        trackedObject (TrackedObject): _description_
        degree (int, optional): _description_. Defaults to 3.
        historyDepth (int, optional): _description_. Defaults to 3.
        futureDepth (int, optional): _description_. Defaults to 30.
    """
    predictSpline(trackedObject, historyDepth=historyDepth, futureDepth=futureDepth)
    for idx in range(1, len(trackedObject.futureY)-1):
        if ((trackedObject.futureY[idx] > trackedObject.futureY[idx-1] and trackedObject.futureY[idx] > trackedObject.futureY[idx+1]) or \
            (trackedObject.futureY[idx] < trackedObject.futureY[idx-1] and trackedObject.futureY[idx] < trackedObject.futureY[idx+1]) or \
            trackedObject.futureY[idx] < 0):
            predictLinear(trackedObject, futureDepth=futureDepth, historyDepth=historyDepth)
            break

def draw_predictions(trackedObject: TrackedObject, image, frameNumber):
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
                cv.circle(image, (int(x), int(y)), 2, color=(0,0,255))
            idx += 1

def draw_history(trackedObject, image, frameNumber):
    if trackedObject.history[-1].frameID == frameNumber:
        idx = 0
        for det in trackedObject.history:
            if (idx % 1) == 0:
                cv.circle(image, (int(det.X), int(det.Y)), 1, color=(0,255,0))
            idx += 1

def predictThread(trackedObject : TrackedObject, image : np.ndarray, frameNumber : int, historyDepth=3, futureDepth=30) -> None:
    if trackedObject.isMoving:
        draw_history(trackedObject, image, frameNumber)
        predictLinPoly(trackedObject, historyDepth, futureDepth)
        draw_predictions(trackedObject, image, frameNumber)
    
def calcWeigths(historyX: np.ndarray, AX: float, historyY: np.ndarray, AY: float):
    weightedX = historyX + AX
    weightedY = historyY + AY
    # for idx in range(np.min(np.array([np.array(historyX).shape[0], np.array(historyY).shape[0]]))):
    #   weightedX[idx] = historyX + AX
    #   weightedX[idx] = historyY + AY
    return weightedX, weightedY

def predictWeightedLinPoly(trackedObject: TrackedObject, historyDepth=3, futureDepth=30):
    """Predict future position of tracked object with weighted training sets.

    Args:
        trackedObject (TrackedObject): The tracked object which future position is to be predicted. 
        historyDepth (int, optional): The length of the training set. Defaults to 3.
        futureDepth (int, optional): The length of the predicted set. Defaults to 30.
    """
    train_X = np.array([det.X for det in trackedObject.history]) 
    train_y= np.array([det.Y for det in trackedObject.history])
    train_X, train_y = calcWeigths(train_X, trackedObject.AX, train_y, trackedObject.AY)
    if len(train_X) >= 3 and len(train_y) >= 3:
        # k (int) : number of training points
        k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in train_X[-historyDepth:-1:slice]])
        y_train = np.array([y for y in train_y[-historyDepth:-1:slice]])
        # generating future points
        if movementIsRight(trackedObject):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        ##### Polynom fitting #####
        polyModel = make_pipeline(PolynomialFeatures(degree=2), linear_model.RANSACRegressor(base_estimator=linear_model.Ridge(alpha=0.5), random_state=30, min_samples=X_train.reshape(-1,1).shape[1]+1))
        polyModel.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
        # print(X_train.shape, y_train.shape)
        y_pred = polyModel.predict(X_test.reshape(-1, 1))
        for idx in range(1, len(y_pred)-1):
            if ((y_pred[idx] > y_pred[idx-1] and y_pred[idx] > y_pred[idx+1]) or \
                (y_pred[idx] < y_pred[idx-1] and y_pred[idx] < y_pred[idx+1]) or \
                (y_pred[idx] < 0) or y_pred[-1] < 0):
                ##### Linear fitting #####
                model = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(n_jobs=-1), random_state=30, min_samples=X_train.reshape(-1,1).shape[1]+1)
                reg = model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
                y_pred = reg.predict(X_test.reshape(-1,1))
                break
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred