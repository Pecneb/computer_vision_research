from utility.dataset import load_dataset
import time
from numpysocket import NumpySocket
import numpy as np
from argparse import ArgumentParser
import yaml
import os
import sys
from multiprocessing import Process




class ComputerVisionResearchServer:
    def __init__(self, conf):
        self.conf = conf
        self.command_lib = {
           "db": self.send_db,
           "video": self.send_video,
        }
        start_t = time.time()
        if os.path.isdir(conf["database"]):
            file_names = os.listdir(conf["database"])
            self.tracks = []
            for i,file_name in enumerate(file_names):
                if file_name.endswith(".joblib"):
                    print(f"{i+1}/{len(file_names)} Loading data from file: {os.path.join(conf['database'], file_name)}")
                    temp_tracks = load_dataset(os.path.join(conf["database"], file_name))
                    self.tracks = np.concatenate((self.tracks, temp_tracks), axis=0)
        else:
            self.tracks = load_dataset(conf["database"])
        print(f"DB loaded under {time.time()-start_t} seconds.")

        self.s = NumpySocket()
        self.s.bind(('127.0.0.1', 9999))
        self.s.listen()
        conn, addr = self.s.accept()
        while True:
            # command = conn.recv()
            # print(commands)
            request = conn.recv(1024)
            request = request.decode("utf-8")
            print(request)
            conn.send("closed".encode("utf-8"))
            # self.send_db(conn)
            conn.close()

    def send_db(self, conn):
        start_t = time.time()
        conn.sendall(self.tracks)
        print(f"DB sended under {time.time()-start_t} seconds")

    def send_video(self, conn):
        conn.send(self.conf["video"])
    
def db_server(conf):
    start_t = time.time()
    if os.path.isdir(conf["database"]):
        file_names = os.listdir(conf["database"])
        tracks = []
        for i,file_name in enumerate(file_names):
            if file_name.endswith(".joblib"):
                print(f"{i+1}/{len(file_names)} Loading data from file: {os.path.join(conf['database'], file_name)}")
                temp_tracks = load_dataset(os.path.join(conf["database"], file_name))
                tracks = np.concatenate((tracks, temp_tracks), axis=0)
    else:
        tracks = load_dataset(conf["database"])
    print(f"DB loaded under {time.time()-start_t} seconds.")

    s = NumpySocket()
    s.bind(('127.0.0.1', 9999))
    s.listen()
    while True:
        conn, addr = s.accept()
        command = conn.recv()
        command_lib[command]()
        start_t = time.time()
        print(f"DB sended under {time.time()-start_t} seconds")


def main():
    conf = yaml.safe_load(open("./computer_vision_research/config.yml"))
    print("\n\nCONFIG:")
    yaml.dump(conf, sys.stdout, default_flow_style=False)
    print("\n\n")
    # db_server(conf)

    db_server = ComputerVisionResearchServer(conf)



if __name__ == "__main__":
    main()