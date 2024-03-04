import matplotlib.pyplot as plt
import numpy as np
from processing_utils import load_dataset, filter_trajectories
from dataManagementClasses import TrackedObject
from typing import List
from time import perf_counter
from tqdm import tqdm

DATASET = "/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_NE8th_24h_v2/Bellevue_Bellevue_NE8th__2017-09-11_13-08-32.joblib"

if __name__ == "__main__":
    t0 = perf_counter()
    dataset: List[TrackedObject] = load_dataset(DATASET)
    preprocessed_dataset: List[TrackedObject] = filter_trajectories(
        trackedObjects=dataset,
        threshold=0.7,
        enter_exit_dist=0.4,
        detectionDistanceFiltering=False,
        detDist=0.05
    )
    print(f"Loaded dataset in {perf_counter() - t0}")
    X_0 = []
    Y_0 = []
    X_1 = []
    Y_1 = []
    for d in tqdm(preprocessed_dataset, desc="Getting XY coordinates."):
        X_0.append(d.history_X[0])
        X_1.append(d.history_X[-1])
        Y_0.append(d.history_Y[0])
        Y_1.append(d.history_Y[-1])
    plt.scatter(x=X_0, y=Y_0, s=0.1, c="g")
    plt.scatter(x=X_1, y=Y_1, s=0.1, c="r")
    plt.grid(visible=True)
    plt.show()