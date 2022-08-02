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
from databaseLogger import getConnection, closeConnection
from sqlite3 import Error

LOAD_OBJECTS_SCRIPT = """SELECT * FROM objects"""

LOAD_DETECTIONS_SCRIPT = """SELECT detections.objID, frameNum, label, confidence, x, y, width, height, vx, vy, ax, ay
                            FROM detections, objects 
                            WHERE detections.objID = objects.objID"""

LOAD_PREDICTIONS_SCRIPT = """SELECT * FROM detections"""

LOAD_METADATA_SCRIPT = """SELECT * FROM metadata"""

LOAD_REGRESSION_SCRIPT = """SELECT * FROM regression"""

def loadObjects(path2db: str) -> list:
    conn = getConnection(path2db)
    try:
        cur = conn.cursor()
        data = cur.execute(LOAD_OBJECTS_SCRIPT)
        return data
    except Error as e:
        print(e)
    closeConnection(conn)
     

def loadDetections(path2db: str) -> list:
    conn = getConnection(path2db)
    try:
        cur = conn.cursor()
        data = cur.execute(LOAD_DETECTIONS_SCRIPT).fetchall()
        return data
    except Error as e:
        print(e)
    closeConnection(conn)

def loadPredictions(path2db: str) -> list:
    conn = getConnection(path2db)
    try:
        cur = conn.cursor()
        data = cur.execute(LOAD_PREDICTIONS_SCRIPT).fetchall()
        return data
    except Error as e:
        print(e)
    closeConnection(conn)

def loadMetadata(path2db: str) -> list:
    conn = getConnection(path2db)
    try:
        cur = conn.cursor()
        data = cur.execute(LOAD_METADATA_SCRIPT).fetchall()
        return data
    except Error as e:
        print(e)
    closeConnection(conn)

def loadRegression(path2db: str) -> list:
    conn = getConnection(path2db)
    try:
        cur = conn.cursor()
        data = cur.execute(LOAD_REGRESSION_SCRIPT).fetchall()
        return data
    except Error as e:
        print(e)
    closeConnection(conn)

