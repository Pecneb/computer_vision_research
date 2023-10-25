import os

import numpy as np


def checkDir(path2db):
    """Check for dir of given database, to be able to save plots.

    Args:
        path2db (str): Path to database. 
    """
    if not os.path.isdir(os.path.join("research_data", path2db.split('/')[-1].split('.')[0])):
        os.mkdir(os.path.join("research_data",
                 path2db.split('/')[-1].split('.')[0]))
        print(
            "Directory \"research_data/{}\" is created.".format(path2db.split('/')[-1].split('.')[0]))


def diff(x_1: float, x_2: float, dt: float) -> float:
    """Differentiate with function x_(i+1) - x_i / dt

    Args:
        x_1 (float): x_i 
        x_2 (float): x_(i+1) 
        dt (float): dt 

    Returns:
        float: dx
    """
    if dt == 0:
        return 0
    return (x_2-x_1) / dt


def dt(t1: float, t2: float) -> float:
    """Calculate dt

    Args:
        t1 (float): t_i 
        t2 (float): t_(i+1) 

    Returns:
        float: dt 
    """
    return t2-t1


def diffmap(a: np.array, t: np.array, k: int):
    """Differentiate an array `a` with time vector `t`, and `k` i+k in the function x_(i+k) - x_i / t_(i+k) - t_i

    Args:
        a (np.array): array of values to differentiate 
        t (np.array): times to differentiate with 
        k (int): stepsize 

    Returns:
        np.array, np.array: Return dX and t timestamps of dX with the logic dx_i, t_i+k 
    """
    X = np.array([])
    T = np.array([])
    if a.shape[0] < k:
        for i in range(a.shape[0]-1):
            T = np.append(T, [t[i]])
            X = np.append(X, [0])
    else:
        for i in range(0, k-1):
            T = np.append(T, [t[i]])
            X = np.append(X, [0])
        for i in range(k, a.shape[0]):
            dt_ = dt(t[i], t[i-k])
            T = np.append(T, t[i])
            X = np.append(X, diff(a[i], a[i-k], dt_))
    return X, T


def strfy_dict_params(params: dict):
    """Stringify params stored in dictionaries.

    Args:
        params (dict): Dict storing the params. 

    Returns:
        str: Stringified params returned in the format "_param1_value1_param2_value2". 
    """
    ret_str = ""
    if len(params) == 0:
        return ret_str
    for p in params:
        ret_str += str("_"+p+"_"+str(params[p]))
    return ret_str
