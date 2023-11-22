from multiprocessing.connection import Client
import joblib
import time
import zmq
import numpy as np

import logging

from numpysocket import NumpySocket

logger = logging.getLogger("simple server")
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    start_t = time.time()
    s.connect(("localhost", 9999))
    s.send("Hello".encode("utf-8")[:1024])
    response = s.recv(1024)
    response = response.decode("utf-8")
    print(response)
    exit(0)
    # asd = s.recv()
    # s.sendall(np.array([1,2,3]))
    # logger.info("sending numpy array:")
    # frame = np.arange(10)
    # s.sendall(frame)
    tracks = s.recv()
    # video = s.recv()
    print("Getting time: ", time.time()-start_t)
    print(tracks[:10])
    # print(video)
