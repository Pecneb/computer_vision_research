import yaml
import sys
import numpy as np
from time import sleep
from multiprocessing import shared_memory
from typing import Dict, Any

sys.path.append("/media/pecneb/970evoplus/gitclones/computer_vision_research")
from trajectorynet.utility.dataset import load_shared_dataset

if __name__ == "__main__":
    with open("config/data_server.yaml", "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file) 
    dataset = load_shared_dataset(config)
    print(dataset)