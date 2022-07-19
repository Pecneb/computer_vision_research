import sqlite3
from sqlite3 import Error
import os

INSERT_SCRIPT = """INSERT INTO object (objID, frameNum, x, y, widht, height)
                    VALUES(?,?,?,?,?,?)"""

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
    try:
        conn = sqlite3.connect(os.path.join("research_data", video_name, db_name))
        print("SQLite version %s", sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS objects (
                                objID INTEGER NOT NULL UNIQUE,
                                frameNum INTEGER NOT NULL,
                                confidence REAL NOT NULL,
                                x REAL NOT NULL,
                                y REAL NOT NULL,
                                width REAL NOT NULL,
                                height REAL NOT NULL
                            );""")

def getConnection(db_name):
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        print("Database connection created!")
        return conn
    except Error as e:
        print(e)
    
    return conn

def logDetection(db_name, objID, frameNum, confidence, x, y, width, height):
    conn = getConnection(db_name)

    try:
        cur = conn.cursor()
        cur.execute(INSERT_SCRIPT, (objID, frameNum, confidence, x, y, width, height))
        conn.commit()
        conn.close()
    except Error as e:
        print(e)

