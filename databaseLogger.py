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
import sqlite3
from sqlite3 import Error
import os
import numpy

INSERT_METADATA = """INSERT INTO metadata (historyDepth, futureDepth, yoloVersion, device, imgsize, stride, confidence_threshold, iou_threshold)
                    VALUES(?,?,?,?,?,?,?,?)"""

INSERT_OBJECT = """INSERT INTO objects (objID, label) VALUES(?,?)"""

INSERT_DETECTION = """INSERT INTO detections (objID, frameNum, confidence, x, y, width, height, vx, vy, ax, ay)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)"""

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
                                yoloVersion INTEGER NOT NULL,   
                                device TEXT NOT NULL,
                                imgsize INTEGER NOT NULL,
                                stride INTEGER NOT NULL,
                                confidence_threshold REAL NOT NULL,
                                iou_threshold REAL NOT NULL
                            );"""

QUERY_LASTFRAME = """SELECT frameNum
                     FROM detections
                     ORDER BY frameNum DESC
                     LIMIT 1"""

def bbox2float(img0: numpy.ndarray, x: int, y: int, w: int, h: int, vx: float, vy: float, ax: float, ay: float):
    """Downscale img0 bbox values to number between 0.0 - 1.0. 
    The output of this function should be the input of logDetection().
    Downscales coordinates, bbox width and height to floats with img0.shape.

    Args:
        x (int): center x coord
        y (int): center y coord
        w (int): width of bbox
        h (int): height of bbox
        vx (float) : velocity x
        vy (float) : velocity y
        
    Return:
        return x, y, w, h, vx, vy
    """
    return x / img0.shape[1], y / img0.shape[0],  w / img0.shape[1], h / img0.shape[0], vx / img0.shape[1], vy / img0.shape[0], ax / img0.shape[1], ay / img0.shape[0]

def prediction2float(img0: numpy.ndarray, x: float, y: float):
    """Downscale prediction coordinates to floats.

    Args:
        img0 (numpy.ndarray): Actual frame when prediction occured.
        x (int): X coordinate
        y (int): Y coordinate
    """
    return x / img0.shape[1], y / img0.shape[0] 

def init_db(video_name: str):
    """Initialize SQLite3 database. Input video_name which is the DIR name.
    DB_name will be the name of the database. If directory does not exists,
    then create one. Creates database from given schema.

    Args:
        video_name (str): The video source's name is the dir name.
        db_name (str): Database name.
    """
    if not os.path.isdir(os.path.join("research_data", video_name)):
        # chekc if directory already exists, if not create one
        os.mkdir(os.path.join("research_data", video_name))
    db_name = video_name + ".db" # database name is the video name with .db appended at the end
    db_path = os.path.join("research_data", video_name, db_name)
    try:
        conn = getConnection(db_path)
        print("SQLite version %s", sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.executescript(SCHEMA)
            conn.commit()
            conn.close()
            print("Detections table created!")

def getConnection(db_path: str):
    """Creates connection to the database.

    Args:
        db_name (str): path to the database file.

    Returns:
        sqlite3.Connection: sqlite3 Connection object.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        print("Database connection created!")
        return conn
    except Error as e:
        print(e)
    
    return conn

def closeConnection(conn : sqlite3.Connection):
    """Closes connection to the database.

    Args:
        conn (sqlite3.Connection): Database connection, that is going to be closed.
    """
    try:
        conn.close()
        print("Database connection closed!")
    except Error as e:
        print(e)
def logDetection(conn : sqlite3.Connection, img0: numpy.ndarray, objID: int, frameNum:int, 
    confidence: float, x_coord: int, y_coord: int, width: int, height: int, x_vel: float, y_vel: float, x_acc: float, y_acc: float):
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
            x, y, w, h, vx, vy, ax, ay = bbox2float(img0, x_coord, y_coord, width, height, x_vel, y_vel, x_acc, y_acc)
            cur.execute(INSERT_DETECTION, (objID, frameNum, confidence, x, y, w, h, vx, vy, ax, ay))
            conn.commit()
            # print("Detection added to database.")
        except Error as e:
            print(e)
        
def objExists(conn: sqlite3.Connection, objID: int, label: str):
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

def logPredictions(conn: sqlite3.Connection, img0: numpy.ndarray, objID:int, frameNumber: int, x: numpy.ndarray, y: numpy.ndarray):
    """Log predictions to database, in format (objID, frameNum, idx, x, y)

    Args:
        conn (sqlite3.Connection): Connection to the database. 
        x (int): X coordinate prediction
        y (int): Y coordinate prediction
    """
    try:
        cur = conn.cursor()
        for idx in range(len(x)):
            x_pred, y_pred = prediction2float(img0, x[idx], y[idx])
            cur.execute(INSERT_PREDICTION, (objID, frameNumber, idx, x_pred, y_pred))
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
    """Logs environment data to the database.

    Args:
        conn (sqlite3.Connection): Connection to the database. 
        historyDepth (int): Length of stored detection history in the memory 
        futureDepth (int): Length of prediction vector. 
    """
    try:
        cur = conn.cursor()
        cur.execute(INSERT_METADATA, (historyDepth, futureDepth, yoloVersion, device, imsz, stride, conf_thres, iou_thres))
        conn.commit()
    except Error as e:
        print(e)