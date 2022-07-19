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

def getConnection(db_name):
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        print("Database connection created!")
        return conn
    except Error as e:
        print(e)
    
    return conn

def closeConnection(conn : sqlite3.connect):
    try:
        conn.close()
    except Error as e:
        print(e)

def logDetection(conn : sqlite3.Connection, img0, objID, frameNum, label, confidence, x_coord, y_coord, width, height):
    try:
        cur = conn.cursor()
        x, y, w, h = bbox2float(img0, x_coord, y_coord, width, height)
        cur.execute(INSERT_DETECTION, (objID, frameNum, label, confidence, x, y, w, h))
        conn.commit()
        print("Detection added to database.")
    except Error as e:
        print(e)
        
def entryExists(conn: sqlite3.Connection, objID, frameNum):
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