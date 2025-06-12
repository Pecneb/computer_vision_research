import yaml
import numpy as np
from time import sleep
from typing import Dict, Any
from multiprocessing import shared_memory, Process
from utility.dataset import load_dataset, load_shared_dataset


def child_process(config: Dict[str, Any]):
    dataset = load_shared_dataset(config)
    print(dataset[:10])


def main():
    with open("config/data_server.yaml", "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    print("Loading dataset...")
    dataset = load_dataset(config["dataset"])
    print("Dataset loaded.")
    print("Starting server...")
    shm = shared_memory.SharedMemory(create=True, size=dataset.nbytes) 
    shared_dataset = np.ndarray(dataset.shape, dtype=dataset.dtype, buffer=shm.buf)
    shared_dataset[:] = dataset[:]
    print(shared_dataset[:10])
    print(shm.size)
    print(shared_dataset.shape)
    print("Serving dataset in shared memory: ", shm.name)
    config["runtime"] = {}
    config["runtime"]["shm_name"] = shm.name
    config["runtime"]["dataset_shape"] = dataset.shape
    config["runtime"]["dataset_dtype"] = dataset.dtype.str
    config["runtime"]["dataset_nbytes"] = dataset.nbytes
    with open("config/data_server.yaml", "w") as file:
        yaml.safe_dump(config, file)
    print("Press Ctrl+C to exit.")
    try:
        p = Process(target=child_process, args=(config,))
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
    finally:
        shm.close()
        shm.unlink()

if __name__ == "__main__":
    main()