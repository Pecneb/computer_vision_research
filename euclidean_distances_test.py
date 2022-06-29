from sklearn.metrics import euclidean_distances
import numpy as np
from scipy.spatial.distance import euclidean
import time

arr = np.array([[1.0, 3.0],
                [4.0, 3.0]], dtype=np.float32)

start_sk = time.time()
print(euclidean_distances([arr[0]], [arr[1]]))
stop_sk = time.time()
start_sp = time.time()
print(euclidean(arr[0], arr[1]))
stop_sp = time.time()
print("Sklearn distance time: {}".format(stop_sk-start_sk))
print("Scipy distance time: {}".format(stop_sp-start_sp))