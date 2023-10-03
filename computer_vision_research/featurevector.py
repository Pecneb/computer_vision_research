import numpy as np
from typing import List, Tuple
from scipy.signal import savgol_filter

class FeatureVector(object):
    """Class representing a feature vector.
    """
    def __call__(self, version: str="1", **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if version == "1":
            return self.factory_1(trackedObjects=kwargs["trackedObjects"],
                                  labels=kwargs["labels"],
                                  pooled_labels=kwargs["pooled_labels"],
                                  k=kwargs["k"],
                                  up_until=kwargs["up_until"])
        if version == "1SG":
            return self.factory_1_SG(trackedObjects=kwargs["trackedObjects"],
                                  labels=kwargs["labels"],
                                  pooled_labels=kwargs["pooled_labels"],
                                  k=kwargs["k"],
                                  up_until=kwargs["up_until"],
                                  window=kwargs["window"],
                                  polyorder=kwargs["polyorder"])

    def _1(self, detections: List) -> np.ndarray:
        """Version 1 feature vectors.

        Parameters
        ----------
        detections : List
            Consecutive detections creating trajectory.

        Returns
        -------
        ndarray
            Feature vector.
        """
        _size = len(detections)
        _half = _size // 2
        return np.array([detections[0].X, detections[0].Y, 
                           detections[0].VX, detections[0].VY,
                           detections[_half].X, detections[_half].Y,
                           detections[-1].X, detections[-1].Y, 
                           detections[-1].VX, detections[-1].VY])
    
    def _1_SG(self, detections: List, window_length: int=7, polyorder: int=2) -> np.ndarray:
        """Version 1 feature vectors.

        Parameters
        ----------
        detections : List
            Consecutive detections creating trajectory.
        window_lenght : int
            Size of savgol filter window. Default: 7.
        polyorder : int
            The polynom degree of the savgol filter. Default: 2.

        Returns
        -------
        ndarray
            Feature vector.
        """
        _size = len(detections)
        _half = _size // 2
        _X = np.array([detections[i].X for i in range(_size)])
        _X_s = savgol_filter(_X, window_length=window_length, polyorder=polyorder)
        _Y = np.array([detections[i].Y for i in range(_size)])
        _Y_s = savgol_filter(_Y, window_length=window_length, polyorder=polyorder)
        _VX = np.array([detections[i].VX for i in range(_size)])
        _VX_s = savgol_filter(_VX, window_length=window_length, polyorder=polyorder)
        _VY = np.array([detections[i].VY for i in range(_size)])
        _VY_s = savgol_filter(_VY, window_length=window_length, polyorder=polyorder)
        return np.array([detections[0].X, detections[0].Y, 
                           detections[0].VX, detections[0].VY,
                           detections[_half].X, detections[_half].Y,
                           detections[-1].X, detections[-1].Y, 
                           detections[-1].VX, detections[-1].VY])
            
    def factory_1(self, trackedObjects: List, labels: np.ndarray, pooled_labels: np.ndarray, k: int=6, up_until: float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature vectors from the histories of trajectories.
        Divide the trajectory into k parts, then make k number of feature vectors.

        Parameters
        ----------
        trackedObjects : List
            List of trajectories.
        labels : np.ndarray
            List of corresponding labels.
        pooled_labels : np.ndarray
            List of corresponding pooled labels.
        k : int, optional
            Number of subtrajectories, by default 6

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the generated feature vectors, corresponding labels, corresponding pooled labels, corresponding metadata.
        """
        featureVectors = []
        new_labels = []
        new_pooled_labels = []
        track_history_metadata = [] # list of [start_time, mid_time, end_time, history_length, trackID]
        #TODO remove time vector, use track_history_metadata instead
        for j in range(len(trackedObjects)):
            step = len(trackedObjects[j].history)//k
            # step = k
            if step > 0:
                midstep = step//2
                for i in range(0, int(len(trackedObjects[j].history)*up_until)-step, step):
                    # featureVectors.append(np.array([trackedObjects[j].history[i].X, trackedObjects[j].history[i].Y, 
                    #                             trackedObjects[j].history[i].VX, trackedObjects[j].history[i].VY,
                    #                             trackedObjects[j].history[i+midstep].X, trackedObjects[j].history[i+midstep].Y,
                    #                             trackedObjects[j].history[i+step].X, trackedObjects[j].history[i+step].Y,
                    #                             trackedObjects[j].history[i+step].VX, trackedObjects[j].history[i+step].VY]))
                    featureVectors.append(self._1(trackedObjects[j].history[i:i+step+1]))
                    if labels is not None:
                        new_labels.append(labels[j])
                    if pooled_labels is not None:
                        new_pooled_labels.append(pooled_labels[j])
                    track_history_metadata.append([trackedObjects[j].history[i].frameID, trackedObjects[j].history[i+midstep].frameID, 
                    trackedObjects[j].history[i+step].frameID, len(trackedObjects[j].history), trackedObjects[j]])
        return np.array(featureVectors), np.array(new_labels), np.array(new_pooled_labels), np.array(track_history_metadata) 

    def factory_1_SG(self, trackedObjects: List, labels: np.ndarray, pooled_labels: np.ndarray, k: int=6, up_until: float=1, window: int=7, polyorder: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature vectors from the histories of trajectories.
        Divide the trajectory into k parts, then make k number of feature vectors.
        Apply savgol filter on the coordinates and velocities of the trajectory part.

        Parameters
        ----------
        trackedObjects : List
            List of trajectories.
        labels : np.ndarray
            List of corresponding labels.
        pooled_labels : np.ndarray
            List of corresponding pooled labels.
        k : int, optional
            Number of subtrajectories, by default 6
        window : int, optional
            Window of savgol filter, by default 7
        polyorder : int, optional
            Degree of polynom used in savgol filter, by default 2

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the generated feature vectors, corresponding labels, corresponding pooled labels, corresponding metadata.
        """
        featureVectors = []
        new_labels = []
        new_pooled_labels = []
        track_history_metadata = [] # list of [start_time, mid_time, end_time, history_length, trackID]
        #TODO remove time vector, use track_history_metadata instead
        for j in range(len(trackedObjects)):
            step_calc = len(trackedObjects[j].history)//k
            step = step_calc if step_calc >= window else window
            # step = k
            if step > 0:
                midstep = step//2
                for i in range(0, int(len(trackedObjects[j].history)*up_until)-step, step):
                    # featureVectors.append(np.array([trackedObjects[j].history[i].X, trackedObjects[j].history[i].Y, 
                    #                             trackedObjects[j].history[i].VX, trackedObjects[j].history[i].VY,
                    #                             trackedObjects[j].history[i+midstep].X, trackedObjects[j].history[i+midstep].Y,
                    #                             trackedObjects[j].history[i+step].X, trackedObjects[j].history[i+step].Y,
                    #                             trackedObjects[j].history[i+step].VX, trackedObjects[j].history[i+step].VY]))
                    featureVectors.append(self._1_SG(trackedObjects[j].history[i:i+step+1], window_length=window, polyorder=polyorder))
                    if labels is not None:
                        new_labels.append(labels[j])
                    if pooled_labels is not None:
                        new_pooled_labels.append(pooled_labels[j])
                    track_history_metadata.append([trackedObjects[j].history[i].frameID, trackedObjects[j].history[i+midstep].frameID, 
                    trackedObjects[j].history[i+step].frameID, len(trackedObjects[j].history), trackedObjects[j]])
        return np.array(featureVectors), np.array(new_labels), np.array(new_pooled_labels), np.array(track_history_metadata) 