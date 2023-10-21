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
from computer_vision_research.utility.dataManagementClasses import TrackedObject


def calcDist(prev, act):
    """Function to calculate distance between an object on previous frame and actual frame.

    Args:
        prev (Detection): Object from previous frame
        act (Detection): Object from actual frame

    Returns:
        xDist, yDist: distance of x coordiantes and distance of y coordiantes
    """
    xDist = abs(prev.X-act.X)
    yDist = abs(prev.Y-act.Y)
    return xDist, yDist

def updateHistory(detections, history, frameNumber, historyDepth=3, disThresh=0.05):
    """Function to update detection history

    Args:
        detections (list[Detection]): a list of new detection
        history (list[TrackedObject]): the tracking history
        frameNumber (int): number of the current video frame
        historyDepth (int): length of the history to be stored
        thresh (float, optional): Threshold to be able to tell if next obj is already detected or is a new one. Defaults to 0.1.
    """
    for next in detections:
        added = False
        for objHistory in history:
            try:
                prev = objHistory.history[-historyDepth]
            except:
                prev = objHistory.history[-1]
            xDist, yDist = calcDist(prev, next)
            if  (xDist < (prev.X * disThresh)) and (yDist < (prev.Y * disThresh)) and objHistory.label == next.label:
                objHistory.history.append(next)
                added = True
                # the threshold for the non moving objects is still harcoded
                # TODO: find a good way to tell what objects are still or in motion
                if (xDist > (disThresh * prev.X * 0.25)) or (yDist > (disThresh * prev.Y * 0.25)):
                    # print("ObjID: {} with xDist: {} and yDist: {} is moving".format(objHistory.objID, xDist, yDist))
                    objHistory.isMoving = True
                else:
                    # print("ObjID: {} with xDist: {} and yDist: {} is not moving".format(objHistory.objID, xDist, yDist))
                    objHistory.isMoving = False
            # remove objects that are older than frameNumber-historyDepth
            if objHistory.history[-1].frameID < (frameNumber-historyDepth):
                try:
                    print("ID {} object is removed from history".format(objHistory.objID))
                    history.remove(objHistory)
                except:
                    continue
        if not added:
            history.append(TrackedObject(len(history)+1, next))