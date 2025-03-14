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
import os
import sqlite3
from pathlib import Path
from sqlite3 import Error
from string import ascii_uppercase

import numpy

INSERT_METADATA = """INSERT INTO metadata (historyDepth, futureDepth, yoloVersion, device, imgsize, stride, confidence_threshold, iou_threshold, k_velocity, k_acceleration)
                    VALUES(?,?,?,?,?,?,?,?,?,?)"""

INSERT_REGRESSION = """INSERT INTO regression (linearFunction, polynomFunction, polynomDegree, trainingPoints)
                    VALUES(?,?,?,?)"""

INSERT_OBJECT = """INSERT INTO objects (objID, label) VALUES(?,?)"""

INSERT_DETECTION = """INSERT INTO detections (objID, frameNum, confidence, x, y, width, height, vx, vy, ax, ay, vx_c, vy_c, ax_c, ay_c)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

INSERT_PREDICTION = """INSERT INTO predictions (objID, frameNum, idx, x, y)
                    VALUES(?,?,?,?,?)"""

QUERY_OBJ = """SELECT objID, label 
                 FROM objects 
                 WHERE objID=? AND label=?"""

QUERY_DETECTION = """SELECT objID, frameNum
                      FROM detections
                      WHERE objID=? AND frameNum=?"""

SCHEMA = """CREATE TABLE IF NOT EXISTS objects (
                                objID INTEGER PRIMARY KEY NOT NULL,
                                label TEXT NOT NULL
                            );
            CREATE TABLE IF NOT EXISTS detections (
                                objID INTEGER NOT NULL,
                                frameNum INTEGER NOT NULL,
                                confidence REAL NOT NULL,
                                x REAL NOT NULL,
                                y REAL NOT NULL,
                                width REAL NOT NULL,
                                height REAL NOT NULL,
                                vx REAL NOT NULL,
                                vy REAL NOT NULL,
                                ax REAL NOT NULL,
                                ay REAL NOT NULL,
                                vx_c REAL NOT NULL,
                                vy_c REAL NOT NULL,
                                ax_c REAL NOT NULL,
                                ay_c REAL NOT NULL,
                                FOREIGN KEY(objID) REFERENCES objects(objID)
                            );
            CREATE TABLE IF NOT EXISTS predictions (
                                objID INTEGER NOT NULL,
                                frameNum INTEGER NOT NULL,
                                idx INTEGER NOT NULL,
                                x REAL NOT NULL,
                                y REAL NOT NULL
                            );
            CREATE TABLE IF NOT EXISTS metadata (
                                historyDepth INTEGER NOT NULL,
                                futureDepth INTEGER NOT NULL,
                                yoloVersion TEXT NOT NULL,   
                                device TEXT NOT NULL,
                                imgsize INTEGER NOT NULL,
                                stride INTEGER NOT NULL,
                                confidence_threshold REAL NOT NULL,
                                iou_threshold REAL NOT NULL,
                                k_velocity REAL NOT NULL,
                                k_acceleration REAL NOT NULL
                            );
            CREATE TABLE IF NOT EXISTS regression (
                                linearFunction TEXT NOT NULL,
                                polynomFunction TEXT NOT NULL,
                                polynomDegree INTEGER NOT NULL,
                                trainingPoints INTEGER NOT NULL
                            );"""

QUERY_LASTFRAME = """SELECT frameNum
                     FROM detections
                     ORDER BY frameNum DESC
                     LIMIT 1"""


def dowscalecoord(scalenum: float, coord: float, a: float):
    """Downscale coordinate number with given scale number and aspect ratio.

    Args:
        scalenum (float): number to scale the coordinate number with
        coord (float): coordinate 
        a (float): aspect ratio 

    Returns:
        float: downscaled value
    """
    return ((coord / scalenum) * a)


def bbox2float(img0: numpy.ndarray, x: int, y: int, w: int, h: int, vx: float, vy: float, ax: float, ay: float, vx_c: float, vy_c: float, ax_c: float, ay_c: float):
    """Downscale bounding box values to with given image's width and height using aspect ratio.

    Args:
        img0 (numpy.ndarray): 3D numpy array containing BGR values shape(height, width, 3)
        x (int): coordinate x 
        y (int): coordinate y 
        w (int): width of the bounding box 
        h (int): height of the bounding box 
        vx (float): velocity of dimesion x 
        vy (float): velocity of dimesion y
        ax (float): acceleration of dimension x 
        ay (float): acceleration of dimension y
        vx_c (float): differentiation of x
        vy_c (float): differentiation of y
        ax_c (float): differentitation if vx_c 
        ay_c (float): differentiation of vy_c

    Returns:
        tuple: tuple of downscaled values 
    """
    aspect_ratio = img0.shape[1] / img0.shape[0]
    rettup = (dowscalecoord(img0.shape[1], x, aspect_ratio),
              dowscalecoord(img0.shape[0], y, 1),
              dowscalecoord(img0.shape[1], w, aspect_ratio),
              dowscalecoord(img0.shape[0], h, 1),
              dowscalecoord(img0.shape[1], vx, aspect_ratio),
              dowscalecoord(img0.shape[0], vy, 1),
              dowscalecoord(img0.shape[1], ax, aspect_ratio),
              dowscalecoord(img0.shape[0], ay, 1),
              dowscalecoord(img0.shape[1], vx_c, aspect_ratio),
              dowscalecoord(img0.shape[0], vy_c, 1),
              dowscalecoord(img0.shape[1], ax_c, aspect_ratio),
              dowscalecoord(img0.shape[0], ay_c, 1))
    return rettup


def prediction2float(img0: numpy.ndarray, x: float, y: float):
    """Downscale prediction coordinates to floats.

    Args:
        img0 (numpy.ndarray): Actual frame when prediction occured.
        x (int): X coordinate
        y (int): Y coordinate
    """
    aspect_ratio = img0.shape[1] / img0.shape[0]
    return (x / img0.shape[1]) * aspect_ratio, y / img0.shape[0]


def init_db(outpath: str or Path):
    """Initialize SQLite3 database. Input video_name which is the DIR name.
    DB_name will be the name of the database. If directory does not exists,
    then create one. Creates database from given schema.

    Args:
        video_name (str): The video source's name is the dir name.
        db_name (str): Database name.
    """
    try:
        conn = getConnection(outpath)
        print("SQLite version %s", sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.executescript(SCHEMA)
            conn.commit()
            conn.close()
            print("Detections table created!")
    return outpath


def getConnection(db_path: str) -> sqlite3.Connection:
    """Creates connection to the database.

    Args:
        db_name (str): path to the database file.

    Returns:
        sqlite3.Connection: sqlite3 Connection object.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except Error as e:
        print(e)

    return conn


def closeConnection(conn: sqlite3.Connection):
    """Closes connection to the database.

    Args:
        conn (sqlite3.Connection): Database connection, that is going to be closed.
    """
    try:
        conn.close()
    except Error as e:
        print(e)


def logDetection(
        conn: sqlite3.Connection, img0: numpy.ndarray, objID: int, frameNum: int,
        confidence: float, x_coord: int, y_coord: int, width: int, height: int,
        x_vel: float, y_vel: float, x_acc: float, y_acc: float, x_vel_calc: float,
        y_vel_calc: float, x_acc_calc: float, y_acc_calc: float
):
    """Logging detections to the database. Downscale bbox coordinates to floats. 
    Insert entry to database if there is no similar entry found.

    Args:
        conn (sqlite3.Connection): Database connection, where data will be logged.
        img0 (numpy matrix): actual image where detections happened.
        objID (int): track id of detection

        frameNum (int): number of actual frame
        label (str): The label of the detected object.
        confidence (float): Confidence of the detection.
        x_coord (int): X center coord of bbox
        y_coord (int): Y center coord of bbox
        width (int): Widht of bbox.
        height (int): Height of bbox.
    """
    if not detectionExists(conn, objID, frameNum):
        try:
            cur = conn.cursor()
            x, y, w, h, vx, vy, ax, ay, vx_c, vy_c, ax_c, ay_c = bbox2float(
                img0, x_coord, y_coord, width, height, x_vel, y_vel, x_acc, y_acc, x_vel_calc, y_vel_calc, x_acc_calc, y_acc_calc)
            cur.execute(INSERT_DETECTION, (objID, frameNum, confidence,
                        x, y, w, h, vx, vy, ax, ay, vx_c, vy_c, ax_c, ay_c))
            conn.commit()
            # print("Detection added to database.")
        except Error as e:
            print(e)


def objExists(conn: sqlite3.Connection, objID: int, label: str) -> bool:
    """Check if entry already exists in database. 
    No multiple detections can occur in a single frame, that have the same objID.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        objID (int): ID of tracked object.
        frameNum (int): Frame in the detection happened.

    Returns:
        bool: True if entry already exists.
    """
    try:
        cur = conn.cursor()
        cur.execute(QUERY_OBJ, (objID, label))
        entry = cur.fetchone()
        if entry is not None:
            return True
        else:
            return False
    except Error as e:
        print(e)


def detectionExists(conn: sqlite3.Connection, objID: int, frameNumber: int):
    """Check if entry already exists in database. 
    No multiple detections can occur in a single frame, that have the same objID.

    Args:
        conn (sqlite3.Connection): Connection to the database.
        objID (int): ID of tracked object.
        frameNum (int): Frame in the detection happened.

    Returns:
        bool: True if entry already exists.
    """
    try:
        cur = conn.cursor()
        cur.execute(QUERY_DETECTION, (objID, frameNumber))
        entry = cur.fetchone()
        if entry is not None:
            return True
        else:
            return False
    except Error as e:
        print(e)


def getLatestFrame(conn: sqlite3.Connection):
    """Gets the number of last frame, when the last sessions logging was stopped.

    Args:
        conn (sqlite3.Connection): Connection to the database. 
    """
    try:
        cur = conn.cursor()
        cur.execute(QUERY_LASTFRAME)
        data = cur.fetchone()
        if data is not None:
            return data[0]
        return False
    except Error as e:
        print(e)


def logPredictions(conn: sqlite3.Connection, img0: numpy.ndarray, objID: int, frameNumber: int, x: numpy.ndarray, y: numpy.ndarray):
    """Log predictions to database, in format (objID, frameNum, idx, x, y)

    Args:
        conn (sqlite3.Connection): Connection to the database. 
        x (int): X coordinate prediction
        y (int): Y coordinate prediction
    """
    try:
        cur = conn.cursor()
        predictions2log = []
        for idx in range(len(x)):
            x_pred, y_pred = prediction2float(img0, x[idx], y[idx])
            predictions2log.append((objID, frameNumber, idx, x_pred, y_pred))
        cur.executemany(INSERT_PREDICTION, predictions2log)
        conn.commit()
    except Error as e:
        print(e)


def logObject(conn: sqlite3.Connection, objID: int, label: str):
    """Log new object to the database.

    Args:
        conn (sqlite3.Connection): Connection to the database. 
        objID (int): Unique id of the new track. 
        label (str): The type of an object, ex. car, person, etc... 
    """
    if not objExists(conn, objID, label):
        try:
            cur = conn.cursor()
            cur.execute(INSERT_OBJECT, (objID, label))
            conn.commit()
        except Error as e:
            print(e)


def logMetaData(conn: sqlite3.Connection, historyDepth: int, futureDepth: int,
                yoloVersion: str, device: str, imsz: int, stride: int, conf_thres: float, iou_thres: float):
    """Log environment data to the database.

    Args:
        conn (sqlite3.Connection): Connection to the database. 
        historyDepth (int): Length of stored detection history in the memory 
        futureDepth (int): Length of prediction vector. 
    """
    try:
        cur = conn.cursor()
        cur.execute(INSERT_METADATA, (historyDepth, futureDepth,
                    yoloVersion, device, imsz, stride, conf_thres, iou_thres))
        conn.commit()
    except Error as e:
        print(e)
        print("SQL error at metadata logging.")


def logRegression(conn: sqlite3.Connection, linearFunction: str, polynomFunction: str, polynomDegree: int, trainingPoints: int):
    """Log regression function data to database.

    Args:
        conn (sqlite3.Connection): Connection to the database. 
        linearFunction (str): Name of the linear regression function. 
        polynomFunction (str): Name of the polynomial regression function. 
        polynomDegree (int): Degree of polynomial features. 
        trainingPoints (int): Number of training points. 
    """
    try:
        cur = conn.cursor()
        cur.execute(INSERT_REGRESSION, (linearFunction,
                    polynomFunction, polynomDegree, trainingPoints))
        conn.commit()
    except Error as e:
        print(e)


def logBuffer(conn: sqlite3.Connection, frame: numpy.ndarray, buffer: list):
    """Log buffer to the database after main loop is ended.

    Args:
        conn (sqlite3.Connection): connection object to the database 
        frame (np.ndarray): frame 
        buffer (list): the buffered data to log, 
        in this sepcific scenario the buffer looks like this list[[objID, 
                                                                   obj.history[-1].frameID,
                                                                   obj.history[-1].confidence
                                                                   obj.X, obj.Y,
                                                                   obj.history[-1].Width,
                                                                   obj.history[-1].Height,
                                                                   obj.VX, obj.VY, obj.AX, obj.AY,
                                                                   obj.futureX, obj.futureY]] 
    """
    import tqdm
    for idx in tqdm.tqdm(range(len(buffer)), desc="Do not interrupt! Logging buffer to database!"):
        logDetection(conn, frame,
                     # objID, frameID, confidence
                     buffer[idx][0], buffer[idx][1], buffer[idx][2],
                     # X, Y
                     buffer[idx][3], buffer[idx][4],
                     # W, H
                     buffer[idx][5], buffer[idx][6],
                     # VX, VY, AX, AY
                     buffer[idx][7], buffer[idx][8], buffer[idx][9], buffer[idx][10],
                     # VX_calculated, VY_calculated
                     buffer[idx][11], buffer[idx][12],
                     buffer[idx][13], buffer[idx][14])                               # AX_calculated, AY_calculated
        # logPredictions(conn, frame, buffer[idx][0], buffer[idx][1], buffer[idx][15], buffer[idx][16])


def logBufferSpeedy(conn: sqlite3.Connection, frame: numpy.ndarray, buffer: list):
    """Log buffer to the database after main loop is ended. 
    Faster than logBuffer(), thanks to executemany() function.

    Args:
        conn (sqlite3.Connection): connection object to the database 
        frame (np.ndarray): frame 
        buffer (list): the buffered data to log, 
        in this sepcific scenario the buffer looks like this list[[objID, 
                                                                   obj.history[-1].frameID,
                                                                   obj.history[-1].confidence
                                                                   obj.X, obj.Y,
                                                                   obj.history[-1].Width,
                                                                   obj.history[-1].Height,
                                                                   obj.VX, obj.VY, obj.AX, obj.AY,
                                                                   obj.futureX, obj.futureY]] 
    """
    print("Dont interrupt! Logging buffer to database!")
    detections2log = []
    for buf in buffer:
        x, y, w, h, vx, vy, ax, ay, vx_c, vy_c, ax_c, ay_c = bbox2float(
            frame, buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14])
        detections2log.append(
            (buf[0], buf[1], buf[2], x, y, w, h, vx, vy, ax, ay, vx_c, vy_c, ax_c, ay_c))
    # predictions2log = []
    # for i in range(len(buffer)):
    #    for j in range(len(buffer[i][11])):
    #        x,y = prediction2float(frame, buffer[i][11][j], buffer[i][12][j])
    #        predictions2log.append((buffer[i][0], buffer[i][1], j, x, y))
    try:
        cur = conn.cursor()
        cur.executemany(INSERT_DETECTION, detections2log)
        # cur.executemany(INSERT_PREDICTION, predictions2log)
        conn.commit()
        print("Buffer to database successfully logged!")
    except Error as e:
        print(e)
