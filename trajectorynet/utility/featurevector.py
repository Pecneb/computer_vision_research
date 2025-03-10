from typing import List, Optional, Tuple, Any, Callable

import numpy as np
import tqdm
from icecream import ic
from scipy.signal import savgol_filter
from joblib import Parallel, delayed, Memory
from numba import jit

ic.disable()


# memory = Memory(location="cache", verbose=0)


class FeatureVector(object):
    """Class representing a feature vector."""

    def __call__(
        self, version: str = "1", **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if version == "1":
            return FeatureVector.factory_1(
                trackedObjects=kwargs["trackedObjects"],
                labels=kwargs["labels"],
                pooled_labels=kwargs["pooled_labels"],
                k=kwargs["k"],
                up_until=kwargs["up_until"],
            )
        if version == "1SG":
            return FeatureVector.factory_1_SG(
                trackedObjects=kwargs["trackedObjects"],
                labels=kwargs["labels"],
                pooled_labels=kwargs["pooled_labels"],
                k=kwargs["k"],
                up_until=kwargs["up_until"],
                window=kwargs["window"],
                polyorder=kwargs["polyorder"],
            )

    @staticmethod
    def _1(detections: List) -> np.ndarray:
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
        return np.array(
            [
                detections[0].X,
                detections[0].Y,
                detections[0].VX,
                detections[0].VY,
                detections[_half].X,
                detections[_half].Y,
                detections[-1].X,
                detections[-1].Y,
                detections[-1].VX,
                detections[-1].VY,
            ]
        )

    @staticmethod
    def _1_SY(detections: List, scale_y: float = 1.5) -> np.ndarray:
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
        return np.array(
            [
                detections[0].X,
                detections[0].Y * scale_y,
                detections[0].VX,
                detections[0].VY * scale_y,
                detections[_half].X,
                detections[_half].Y * scale_y,
                detections[-1].X,
                detections[-1].Y * scale_y,
                detections[-1].VX,
                detections[-1].VY * scale_y,
            ]
        )

    @staticmethod
    def _1_SG(
        detections: List, window_length: int = 7, polyorder: int = 2
    ) -> np.ndarray:
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
        _VX_s = savgol_filter(
            _VX, window_length=window_length, polyorder=polyorder, deriv=1
        )
        _VY = np.array([detections[i].VY for i in range(_size)])
        _VY_s = savgol_filter(
            _VY, window_length=window_length, polyorder=polyorder, deriv=1
        )
        return np.array(
            [
                _X_s[0],
                _Y_s[0],
                _VX_s[0],
                _VY_s[0],
                _X_s[_half],
                _Y_s[_half],
                _X_s[-1],
                _Y_s[-1],
                _VX_s[-1],
                _VY_s[-1],
            ]
        )

    @staticmethod
    def _1_SG_fov_transform(
        detections: List, window_length: int = 7, polyorder: int = 2, fps: float = 30.0
    ) -> np.ndarray:
        """Version 1 feature vectors.
        The velocities are calculated and smoothed with Savitzky Goaly filter.

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
        _Y = np.array([detections[i].Y for i in range(_size)])
        _VX = (
            savgol_filter(_X, window_length=window_length, polyorder=polyorder, deriv=1)
            * fps
        )
        _VY = (
            savgol_filter(_Y, window_length=window_length, polyorder=polyorder, deriv=1)
            * fps
        )
        return np.array(
            [
                _X[0],
                _Y[0],
                _VX[0],
                _VY[0],
                _X[_half],
                _Y[_half],
                _X[-1],
                _Y[-1],
                _VX[-1],
                _VY[-1],
            ]
        )

    @staticmethod
    def _7(
        x: np.ndarray,
        y: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Feature vector version 7

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        vx : np.ndarray
            X velocities.
        vy : np.ndarray
            Y velocities.
        weights : Optional[np.ndarray], optional
            Weight vector, by default ...

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if weights is None:
            _weights = np.array([1, 1, 100, 100, 2, 2, 200, 200], dtype=np.float32)
        else:
            _weights = weights
        return (
            np.array(
                [
                    x[0],
                    y[0],
                    vx[0],
                    vy[0],
                    x[-1],
                    y[-1],
                    vx[-1],
                    vy[-1],
                ]
            )
            * _weights
        )

    @staticmethod
    def _7_SG(
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Feature vector version 7

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        vx : np.ndarray
            X velocities.
        vy : np.ndarray
            Y velocities.
        weights : Optional[np.ndarray], optional
            Weight vector, by default ...
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if weights is None:
            _weights = np.array([1, 1, 100, 100, 2, 2, 200, 200], dtype=np.float32)
        else:
            _weights = weights
        x_s = savgol_filter(x, window_length=window_length, polyorder=polyorder)
        y_s = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        vx_s = savgol_filter(
            x, window_length=window_length, polyorder=polyorder, deriv=1
        )
        vy_s = savgol_filter(
            y, window_length=window_length, polyorder=polyorder, deriv=1
        )
        return (
            np.array(
                [
                    x_s[0],
                    y_s[0],
                    vx_s[0],
                    vy_s[0],
                    x_s[-1],
                    y_s[-1],
                    vx_s[-1],
                    vy_s[-1],
                ]
            )
            * _weights
        )

    @staticmethod
    def _7_SG_fov_transform(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
        weights: Optional[np.ndarray] = None,
        fps: float = 30.0,
    ) -> np.ndarray:
        """Feature vector version 7

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2
        weights : Optional[np.ndarray], optional
            Weight vector, by default ...
        fps : float
            Frames per second, by default 30.0

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if weights is None:
            _weights = np.array(
                [1, 1, fps, fps, 2, 2, 2 * fps, 2 * fps], dtype=np.float32
            )
        else:
            _weights = weights
        vx_s = savgol_filter(
            x, window_length=window_length, polyorder=polyorder, deriv=1
        )
        vy_s = savgol_filter(
            y, window_length=window_length, polyorder=polyorder, deriv=1
        )
        return (
            np.array(
                [
                    x[0],
                    y[0],
                    vx_s[0],
                    vy_s[0],
                    x[-1],
                    y[-1],
                    vx_s[-1],
                    vy_s[-1],
                ]
            )
            * _weights
        )

    @staticmethod
    def _8(x: np.ndarray, y: np.ndarray, weight: float = 0.8) -> np.ndarray:
        """Feature vector version 7

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        weight : float
            This weight determines the middle coordinates distance from the first coordinate.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        size = x.shape[0]
        middle_idx = int(size * weight)
        return np.array([x[0], y[0], x[middle_idx], y[middle_idx], x[-1], y[-1]])

    @staticmethod
    def _8_SG(
        x: np.ndarray,
        y: np.ndarray,
        weight: float = 0.8,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Feature vector version 7

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        weight : float
            This weight determines the middle coordinates distance from the first coordinate.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        x_s = savgol_filter(x, window_length=window_length, polyorder=polyorder)
        y_s = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        size = x.shape[0]
        middle_idx = int(size * weight)
        return np.array(
            [x_s[0], y_s[0], x_s[middle_idx], y_s[middle_idx], x_s[-1], y_s[-1]]
        )

    @staticmethod
    def _9(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Feature vector version 9

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        return np.array([x[-1], y[-1]])

    @staticmethod
    def _10(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
        weights: np.ndarray = np.array([1, 1, 100, 100]),
    ) -> np.ndarray:
        """Feature vector version 10, uses only the end coordinates and the end velocities.
        Velocities are calculated and smoothed with Savitzky Goaly filter.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        vx : np.ndarray
            X velocities.
        vy : np.ndarray
            Y velocities.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        vx = savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=1)
        return np.array([x[-1], y[-1], vx[-1], vy[-1]]) * weights

    @staticmethod
    def Re(
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Use only the end coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        return np.array([x[-1], y[-1]])
    
    @staticmethod
    def ReVe(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use only the end coordinates and the end velocities.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        return np.array([x[-1], y[-1], vx[-1]*5, vy[-1]*5]
    )

    @staticmethod
    def ReVeAe(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities and the end accelerations.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
            ax = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=2)
            ay = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=2)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        ax = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        ay = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        return np.array([x[-1], y[-1], vx[-1]*5, vy[-1]*5, ax[-1]*50, ay[-1]*50])
    
    @staticmethod
    def ReRs(
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Use only the end coordinates and the start coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        return np.array([x[-1], y[-1], x[0], y[0]])

    @staticmethod
    def ReVeRs(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities and the start coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        return np.array([x[-1], y[-1], vx[-1]*5, vy[-1]*5, x[0], y[0]])
    
    @staticmethod
    def ReVeAeRs(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities, the end accelerations and the start coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
            ax = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=2)
            ay = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=2)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        ax = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        ay = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        return np.array([x[-1], y[-1], vx[-1]*5, vy[-1]*5, ax[-1]*50, ay[-1]*50, x[0], y[0]])

    @staticmethod
    def ReRm(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use the end coordinates and the middle coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        _size = len(x)
        _half = _size // 2
        return np.array([x[-1], y[-1], x[_half], y[_half]])
    
    @staticmethod
    def ReVeRm(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities and the middle coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        _size = len(x)
        _half = _size // 2
        return np.array([x[-1], y[-1], vx[-1]*5, vy[-1]*5, x[_half], y[_half]])
    
    @staticmethod
    def ReVeAeRm(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities, the end accelerations and the middle coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
            ax = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=2)
            ay = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=2)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        ax = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        ay = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        _size = len(x)
        _half = _size // 2
        return np.array([x[-1], y[-1], vx[-1]*5, vy[-1]*5, ax[-1]*50, ay[-1]*50, x[_half], y[_half]])
    
    @staticmethod
    @jit(nopython=True)
    def ReRsRm(
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Use the end coordinates, the start coordinates and the middle coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        _size = len(x)
        _half = _size // 2
        return np.array([x[-1], y[-1], x[0], y[0], x[_half], y[_half]])
    
    @staticmethod
    def ReVeRsRm(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities, the start coordinates and the middle coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        _size = len(x)
        _half = _size // 2
        return np.array([x[-1], y[-1], vx[-1], vy[-1], x[0], y[0], x[_half], y[_half]]) * np.array([1,1,5,5,1,1,1,1])
    
    @staticmethod
    def ReVeAeRsRm(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 7,
        polyorder: int = 2
    ) -> np.ndarray:
        """Use the end coordinates, the end velocities, the end accelerations, the start coordinates and the middle coordinates.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=1)
            vy = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=1)
            ax = savgol_filter(x, window_length=1, polyorder=polyorder, deriv=2)
            ay = savgol_filter(y, window_length=1, polyorder=polyorder, deriv=2)
        vx = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        vy = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=1)
        ax = savgol_filter(x[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        ay = savgol_filter(y[-window_length:], window_length=window_length, polyorder=polyorder, deriv=2)
        _size = len(x)
        _half = _size // 2
        return np.array([x[-1], y[-1], vx[-1], vy[-1], ax[-1], ay[-1], x[0], y[0], x[_half], y[_half]]) * np.array([1,1,5,5,50,50,1,1,1,1])

    @staticmethod
    def _12(
        x: np.ndarray,
        y: np.ndarray,
        window_length: int = 5,
        polyorder: int = 1,
    ) -> np.ndarray:
        """Feature vector version 11, uses only the end coordinates and the end velocities.
        Velocities are calculated and smoothed with Savitzky Goaly filter.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        window_length : int
            Window size of Savitzky Goaly filter, by default 7
        polyorder : int
            Polynom degree of Savitzky Goaly filter, by default 2

        Returns
        -------
        np.ndarray
            Feature Vector.
        """
        if len(x) < window_length:
            vx = savgol_filter(
                x[-len(x) - 1 :], window_length=1, polyorder=polyorder, deriv=1
            )
            vy = savgol_filter(
                y[-len(y) - 1 :], window_length=1, polyorder=polyorder, deriv=1
            )
            ax = savgol_filter(
                x[-len(x) - 1 :], window_length=1, polyorder=polyorder, deriv=2
            )
            ay = savgol_filter(
                y[-len(y) - 1 :], window_length=1, polyorder=polyorder, deriv=2
            )
            return np.array([x[-1], y[-1], vx[-1], vy[-1], ax[-1], ay[-1]])
        vx = savgol_filter(
            x[-window_length:],
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
        )
        vy = savgol_filter(
            y[-window_length:],
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
        )
        ax = savgol_filter(
            x[-window_length:],
            window_length=window_length,
            polyorder=polyorder,
            deriv=2,
        )
        ay = savgol_filter(
            y[-window_length:],
            window_length=window_length,
            polyorder=polyorder,
            deriv=2,
        )
        return np.array([x[-1], y[-1], vx[-1], vy[-1], ax[-1], ay[-1]])

    @staticmethod
    def factory_1(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        k: int = 6,
        up_until: float = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        # list of [start_time, mid_time, end_time, history_length, trackID]
        track_history_metadata = []
        # TODO remove time vector, use track_history_metadata instead
        for j in range(len(trackedObjects)):
            step = len(trackedObjects[j].history) // k
            # step = k
            if step > 0:
                midstep = step // 2
                for i in range(
                    0, int(len(trackedObjects[j].history) * up_until) - step, step
                ):
                    # featureVectors.append(np.array([trackedObjects[j].history[i].X, trackedObjects[j].history[i].Y,
                    #                             trackedObjects[j].history[i].VX, trackedObjects[j].history[i].VY,
                    #                             trackedObjects[j].history[i+midstep].X, trackedObjects[j].history[i+midstep].Y,
                    #                             trackedObjects[j].history[i+step].X, trackedObjects[j].history[i+step].Y,
                    #                             trackedObjects[j].history[i+step].VX, trackedObjects[j].history[i+step].VY]))
                    featureVectors.append(
                        FeatureVector._1(trackedObjects[j].history[i : i + step + 1])
                    )
                    if labels is not None:
                        new_labels.append(labels[j])
                    if pooled_labels is not None:
                        new_pooled_labels.append(pooled_labels[j])
                    track_history_metadata.append(
                        [
                            trackedObjects[j].history[i].frameID,
                            trackedObjects[j].history[i + midstep].frameID,
                            trackedObjects[j].history[i + step].frameID,
                            len(trackedObjects[j].history),
                            trackedObjects[j],
                        ]
                    )
        return (
            np.array(featureVectors),
            np.array(new_labels),
            np.array(new_pooled_labels),
            np.array(track_history_metadata),
        )

    @staticmethod
    def factory_1_SG(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        k: int = 6,
        up_until: float = 1,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        window_length : int, optional
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
        # list of [start_time, mid_time, end_time, history_length, trackID]
        track_history_metadata = []
        # TODO remove time vector, use track_history_metadata instead
        for j in range(len(trackedObjects)):
            step_calc = len(trackedObjects[j].history) // k
            step = step_calc if step_calc >= window_length else window_length
            # step = k
            if step > 0:
                midstep = step // 2
                for i in range(
                    0, int(len(trackedObjects[j].history) * up_until) - step, step
                ):
                    # featureVectors.append(np.array([trackedObjects[j].history[i].X, trackedObjects[j].history[i].Y,
                    #                             trackedObjects[j].history[i].VX, trackedObjects[j].history[i].VY,
                    #                             trackedObjects[j].history[i+midstep].X, trackedObjects[j].history[i+midstep].Y,
                    #                             trackedObjects[j].history[i+step].X, trackedObjects[j].history[i+step].Y,
                    #                             trackedObjects[j].history[i+step].VX, trackedObjects[j].history[i+step].VY]))
                    featureVectors.append(
                        FeatureVector._1_SG(
                            trackedObjects[j].history[i : i + step + 1],
                            window_length=window_length,
                            polyorder=polyorder,
                        )
                    )
                    if labels is not None:
                        new_labels.append(labels[j])
                    if pooled_labels is not None:
                        new_pooled_labels.append(pooled_labels[j])
                    track_history_metadata.append(
                        [
                            trackedObjects[j].history[i].frameID,
                            trackedObjects[j].history[i + midstep].frameID,
                            trackedObjects[j].history[i + step].frameID,
                            len(trackedObjects[j].history),
                            trackedObjects[j],
                        ]
                    )
        return (
            np.array(featureVectors),
            np.array(new_labels),
            np.array(new_pooled_labels),
            np.array(track_history_metadata),
        )

    @staticmethod
    def factory_1_SY(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        k: int = 6,
        up_until: float = 1,
        scale_y: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature vectors from the histories of trajectories.
        Divide the trajectory into k parts, then make k number of feature vectors.
        Apply scaling to the Y coordinates. This might help to reduce the effect of the camera perspective.

        Parameters
        ----------
        trackedObjects : List
            List of trajectories.
        labels : np.ndarray
            List of corresponding labels.
        pooled_labels : np.ndarray
            List of corresponding pooled labels.
        k : int, optional
            Number of subtrajectories to make, by default 6
        scale_y : float, optional
            Scaling factor, by default 1.5

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the generated feature vectors, corresponding labels, corresponding pooled labels, corresponding metadata.
        """
        featureVectors = []
        new_labels = []
        new_pooled_labels = []
        track_history_metadata = []
        for j in range(len(trackedObjects)):
            step_calc = len(trackedObjects[j].history) // k
            step = step_calc
            if step > 0:
                midstep = step // 2
                for i in range(
                    0, int(len(trackedObjects[j].history) * up_until) - step, step
                ):
                    featureVectors.append(
                        FeatureVector._1_SY(
                            trackedObjects[j].history[i : i + step + 1], scale_y=scale_y
                        )
                    )
                    if labels is not None:
                        new_labels.append(labels[j])
                    if pooled_labels is not None:
                        new_pooled_labels.append(pooled_labels[j])
                    track_history_metadata.append(
                        [
                            trackedObjects[j].history[i].frameID,
                            trackedObjects[j].history[i + midstep].frameID,
                            trackedObjects[j].history[i + step].frameID,
                            len(trackedObjects[j].history),
                            trackedObjects[j],
                        ]
                    )
        return (
            np.array(featureVectors),
            np.array(new_labels),
            np.array(new_pooled_labels),
            np.array(track_history_metadata),
        )

    @staticmethod
    def factory_1_SG_fov_transform(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        k: int = 6,
        up_until: float = 1,
        window_length: int = 7,
        polyorder: int = 2,
        fps: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        up_until : float, optional
            Up until what part of the trajectory to use, by default 1
        window_length : int, optional
            Window of savgol filter, by default 7
        polyorder : int, optional
            Degree of polynom used in savgol filter, by default 2
        fps : float, optional
            Frames per second, by default 30.0

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the generated feature vectors, corresponding labels, corresponding pooled labels, corresponding metadata.
        """
        featureVectors = []
        new_labels = []
        new_pooled_labels = []
        track_history_metadata = []
        for j in range(len(trackedObjects)):
            step_calc = len(trackedObjects[j].history) // k
            step = step_calc if step_calc >= window_length else window_length
            if step > 0:
                midstep = step // 2
                for i in range(
                    0, int(len(trackedObjects[j].history) * up_until) - step, step
                ):
                    featureVectors.append(
                        FeatureVector._1_SG_fov_transform(
                            trackedObjects[j].history[i : i + step + 1],
                            window_length=window_length,
                            polyorder=polyorder,
                            fps=fps,
                        )
                    )
                    if labels is not None:
                        new_labels.append(labels[j])
                    if pooled_labels is not None:
                        new_pooled_labels.append(pooled_labels[j])
        return (
            np.array(featureVectors),
            np.array(new_labels),
            np.array(new_pooled_labels),
            np.array(track_history_metadata),
        )

    # @memory.cache
    @staticmethod
    def factory_7(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int,
        weights: Optional[np.ndarray] = np.array(
            [1, 1, 100, 100, 2, 2, 200, 200], dtype=np.float32
        ),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature vectors from the histories of trajectories.

        Parameters
        ----------
        trackedObjects : List
            List of trajectories.
        labels : np.ndarray
            Labels for each trajectory.
        pooled_labels : np.ndarray
            Pooled labels for each trajectory.
        max_stride : int
            Maximum stride length. This means that the subtrajectory length will be max_stride.
        weights : Optional[np.ndarray], optional
            Weight vector, by default np.array( [1, 1, 100, 100, 2, 2, 200, 200], dtype=np.float32 )

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the generated feature vectors, corresponding labels, corresponding pooled labels, corresponding metadata.
        """
        # weights = np.array([1,1,100,100,2,2,200,200], dtype=np.float32)
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                # midx = j + (3*stride // 4) - 1
                # end_idx = j + stride - 1
                feature_vector = FeatureVector._7(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    vx=t.history_VX_calculated[j : j + max_stride],
                    vy=t.history_VY_calculated[j : j + max_stride],
                    weights=weights,
                )
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                # ic(j,len(trackedObjects[i].history), t.history_X.shape)
                # metadata.append([trackedObjects[i].history[j].frameID, None,
                #     trackedObjects[i].history[end_idx].frameID, len(trackedObjects[i].history), trackedObjects[i]])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def factory_7_fast(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int,
        weights: Optional[np.ndarray] = np.array(
            [1, 1, 100, 100, 2, 2, 200, 200], dtype=np.int16
        ),
        n_jobs: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature vectors from the histories of trajectories.
        Make feature vectors from the whole trajectory, with a stride of max_stride.
        Faster implementation using joblib.

        Parameters
        ----------
        trackedObjects : List
            Tracked objects.
        labels : np.ndarray
            Labels.
        pooled_labels : np.ndarray
            Pooled labels.
        max_stride : int
            Maximum stride length.
        weights : Optional[np.ndarray], optional
            Weight vector, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Feature vectors, corresponding labels and pooled labels.
        """
        # weights = np.array([1,1,100,100,2,2,200,200], dtype=np.float32)
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])

        def append_feature_vectors(
            feature_vector: np.ndarray, label: int, pooled_label: int
        ):
            nonlocal X_feature_vectors, y_new_labels, y_new_pooled_labels
            if X_feature_vectors.shape == (0,):
                X_feature_vectors = np.array(feature_vector).reshape(
                    (-1, feature_vector.shape[0])
                )
            else:
                X_feature_vectors = np.append(
                    X_feature_vectors, np.array([feature_vector]), axis=0
                )
            y_new_labels = np.append(y_new_labels, label)
            y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_label)

        def append_labels(label: int, pooled_label: int):
            nonlocal y_new_labels, y_new_pooled_labels
            y_new_labels = np.append(y_new_labels, label)
            y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_label)

        with Parallel(n_jobs=8, max_nbytes=1e6, require="sharedmem") as parallel:
            for i, t in tqdm.tqdm(
                enumerate(trackedObjects),
                desc="Features for classification.",
                total=len(trackedObjects),
            ):
                stride = max_stride
                if stride > t.history_X.shape[0]:
                    continue
                parallel(
                    delayed(append_feature_vectors)(
                        FeatureVector._7(
                            x=t.history_X[j : j + max_stride],
                            y=t.history_Y[j : j + max_stride],
                            vx=t.history_VX_calculated[j : j + max_stride],
                            vy=t.history_VY_calculated[j : j + max_stride],
                            weights=weights,
                        ),
                        labels[i],
                        pooled_labels[i],
                    )
                    for j in range(0, t.history_X.shape[0] - max_stride, max_stride)
                )
                # parallel(delayed(append_labels)(labels[i], pooled_labels[i]) for _ in range(0, t.history_X.shape[0]-max_stride, max_stride))
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
        )

    @staticmethod
    def factory_7_SG(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int,
        weights: Optional[np.ndarray] = np.array(
            [1, 1, 100, 100, 2, 2, 200, 200], dtype=np.float32
        ),
        window_length: int = 7,
        polyorder: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Factory method for computing feature vectors using 7th order Savitzky-Golay filter.

        Parameters
        ----------
        trackedObjects : List
            List of tracked objects.
        labels : np.ndarray
            Array of labels.
        pooled_labels : np.ndarray
            Array of pooled labels.
        max_stride : int
            Maximum stride.
        weights : Optional[np.ndarray], optional
            Array of weights for each feature, by default np.array([1, 1, 100, 100, 2, 2, 200, 200], dtype=np.float32)
        window_length : int, optional
            Window length for Savitzky-Golay filter, by default 7
        polyorder : int, optional
            Polynomial order for Savitzky-Golay filter, by default 2

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing feature vectors, labels, pooled labels, and metadata.
        """
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                end_idx = j + stride - 1
                feature_vector = FeatureVector._7_SG(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    vx=t.history_VX_calculated[j : j + max_stride],
                    vy=t.history_VY_calculated[j : j + max_stride],
                    weights=weights,
                )
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def factory_7_SG_velocity(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int,
        weights: Optional[np.ndarray] = None,
        window_length: int = 7,
        polyorder: int = 2,
        fps: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Factory method for computing feature vectors using 7th order Savitzky-Golay filter.

        Parameters
        ----------
        trackedObjects : List
            List of tracked objects.
        labels : np.ndarray
            Array of labels.
        pooled_labels : np.ndarray
            Array of pooled labels.
        max_stride : int
            Maximum stride.
        weights : Optional[np.ndarray], optional
            Array of weights for each feature, by default None
        window_length : int, optional
            Window length for Savitzky-Golay filter, by default 7
        polyorder : int, optional
            Polynomial order for Savitzky-Golay filter, by default 2
        fps : float, optional
            Frames per second, by default 30.0

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing feature vectors, labels, pooled labels, and metadata.
        """
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                end_idx = j + stride - 1
                feature_vector = FeatureVector._7_SG_fov_transform(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    window_length=window_length,
                    polyorder=polyorder,
                    fps=fps,
                )
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def factory_8(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int,
        weight: float = 0.8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the feature vectors for a list of tracked objects.

        Parameters
        ----------
        trackedObjects : List
            List of tracked objects.
        labels : np.ndarray
            Array of labels for each tracked object.
        pooled_labels : np.ndarray
            Array of pooled labels for each tracked object.
        max_stride : int
            Maximum stride to be used in the feature vector computation.
        weight : float, optional
            Weight to be used in the feature vector computation, by default 0.8.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing the computed feature vectors, labels, pooled labels, and metadata.
        """
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                end_idx = j + stride - 1
                feature_vector = FeatureVector._8(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    weight=weight,
                )
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def factory_8_SG(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int,
        weight: float = 0.8,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                # midx = j + (3*stride // 4) - 1
                end_idx = j + stride - 1
                feature_vector = FeatureVector._8_SG(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    weight=weight,
                    window_length=window_length,
                    polyorder=polyorder,
                )
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                # ic(j,len(trackedObjects[i].history), t.history_X.shape)
                # metadata.append([trackedObjects[i].history[j].frameID, None,
                #     trackedObjects[i].history[end_idx].frameID, len(trackedObjects[i].history), trackedObjects[i]])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def factory_9(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                feature_vector = FeatureVector._9(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                )
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                # ic(j,len(trackedObjects[i].history), t.history_X.shape)
                # metadata.append([trackedObjects[i].history[j].frameID, None,
                #     trackedObjects[i].history[end_idx].frameID, len(trackedObjects[i].history), trackedObjects[i]])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def factory_10(
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int = 30,
        window_length: int = 7,
        polyorder: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_feature_vectors = np.array([])
        y_new_labels = np.array([])
        y_new_pooled_labels = np.array([])
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                feature_vector = FeatureVector._10(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    window_length=window_length,
                    polyorder=polyorder,
                )
                ic(feature_vector)
                if X_feature_vectors.shape == (0,):
                    X_feature_vectors = np.array(feature_vector).reshape(
                        (-1, feature_vector.shape[0])
                    )
                else:
                    X_feature_vectors = np.append(
                        X_feature_vectors, np.array([feature_vector]), axis=0
                    )
                y_new_labels = np.append(y_new_labels, labels[i])
                y_new_pooled_labels = np.append(y_new_pooled_labels, pooled_labels[i])
                # ic(j,len(trackedObjects[i].history), t.history_X.shape)
                # metadata.append([trackedObjects[i].history[j].frameID, None,
                #     trackedObjects[i].history[end_idx].frameID, len(trackedObjects[i].history), trackedObjects[i]])
                metadata.append([None, None, None, None, trackedObjects[i]])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels),
            np.array(metadata),
        )

    @staticmethod
    def make_feature_vectors(
        func: Callable,
        trackedObjects: List,
        labels: np.ndarray,
        pooled_labels: np.ndarray,
        max_stride: int = 30,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate feature vectors from the histories of trajectories.

        Parameters
        ----------
        trackedObjects : List
            List of trajectories.
        labels : np.ndarray
            List of corresponding labels.
        pooled_labels : np.ndarray
            List of corresponding pooled labels.
        max_stride : int, optional
            Maximum size of trajectory part, by default 30
        window_length : int, optional
            Savitzky Golay window length, by default 5
        polyorder : int, optional
            Savitzky Golay polyorder, by default 1

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Feature vectors with coresponding labels, pooled labels and metadata.
        """
        X_feature_vectors = []
        y_new_labels = []
        y_new_pooled_labels = []
        metadata = []
        for i, t in tqdm.tqdm(
            enumerate(trackedObjects),
            desc="Features for classification.",
            total=len(trackedObjects),
        ):
            stride = max_stride
            if stride > t.history_X.shape[0]:
                continue
            for j in range(0, t.history_X.shape[0] - max_stride, max_stride):
                feature_vector = func(
                    x=t.history_X[j : j + max_stride],
                    y=t.history_Y[j : j + max_stride],
                    **kwargs
                )
                X_feature_vectors.append(feature_vector)
                y_new_labels.append(labels[i])
                y_new_pooled_labels.append(pooled_labels[i])
                metadata.append(trackedObjects[i])
        return (
            np.array(X_feature_vectors),
            np.array(y_new_labels, dtype=int),
            np.array(y_new_pooled_labels, dtype=int),
            np.array(metadata),
        )


def track2features(trackedObject, stride, label=None, pooled_label=None):
    X_feature_vectors = np.array(
        [
            FeatureVector.Re(x=trackedObject.history_X[i : i + stride])
            for i in range(0, len(trackedObject.history_X) - stride, stride)
        ]
    )
    labels = np.array([label for _ in range(0, len(X_feature_vectors))])
    pooled_labels_ = np.array([pooled_label for _ in range(0, len(X_feature_vectors))])
    return X_feature_vectors, labels, pooled_labels_
