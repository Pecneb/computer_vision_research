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

INSERT_DETECTION = """INSERT INTO detections (objID, frameNum, label, confidence, x, y, width, height)
                    VALUES(?,?,?,?,?,?,?,?)"""

QUERY_ENTRY = """SELECT objID, frameNum 
                 FROM detections 
                 WHERE objID=? AND frameNum=?"""

SCHEMA = """CREATE TABLE IF NOT EXISTS detections (
                                objID INTEGER NOT NULL,
                                frameNum INTEGER NOT NULL,
                                label TEXT NOT NULL,
                                confidence REAL NOT NULL,
                                x REAL NOT NULL,
                                y REAL NOT NULL,
                                width REAL NOT NULL,
                                height REAL NOT NULL
                            );"""

QUERY_LASTFRAME = """SELECT frameNum
                     FROM detections
                     ORDER BY frameNum DESC
                     LIMIT 1"""

def bbox2float(img0, x, y, w, h):
    """Downscale img0 bbox to float. 
    The output of this function should be the input of logDetection().
    Downscales coordinates, bbox width and height ot floats with img0.shape.

    Args:
        x (int): center x coord
        y (int): center y coord
        w (int): width of bbox
        h (int): height of bbox
        
    Return:
        return x, y, w, h
    """
    return x / img0.shape[1], y / img0.shape[0],  w / img0.shape[1], h / img0.shape[0]

def init_db(video_name, db_name):
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

def getConnection(db_path):
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

def logDetection(conn : sqlite3.Connection, img0, objID, frameNum, label, confidence, x_coord, y_coord, width, height):
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
    try:
        cur = conn.cursor()
        x, y, w, h = bbox2float(img0, x_coord, y_coord, width, height)
        cur.execute(INSERT_DETECTION, (objID, frameNum, label, confidence, x, y, w, h))
        conn.commit()
        print("Detection added to database.")
    except Error as e:
        print(e)
        
def entryExists(conn: sqlite3.Connection, objID, frameNum):
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
        cur.execute(QUERY_ENTRY, (objID, frameNum))
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
        return data
    except Error as e:
        print(e)