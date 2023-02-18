from processing_utils import (
    tracks2joblib,
    trackslabels2joblib, 
    mergeDatasets,
    downscale_TrackedObjects,
    load_joblib_tracks,
    trackedObjects_old_to_new
)
import argparse
import cv2
from joblib import dump
import numpy as np
from tqdm import tqdm
from dataManagementClasses import TrackedObject
from copy import deepcopy
from itertools import starmap

def mainmodule_function(args): 
    for db in tqdm(args.database, desc="Database converted."):
        tracks2joblib(db, args.n_jobs)

def submodule_function(args):
    trackslabels2joblib(args.database[0], args.min_samples, args.max_eps, args.xi, args.min_cluster_size , args.n_jobs, args.threshold, args.cluster_dimensions)

def submodule_function_2(args):
    mergeDatasets(args.database, args.output)

def submodule_function_3(args):
    trackedObjects = load_joblib_tracks(args.database[0])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError()
    ret, img = cap.read()
    if ret:
        downscale_TrackedObjects(trackedObjects, img)
        dump(trackedObjects, args.database[0], compress="lz4")
    else:
        raise IOError()

def submodule_function_4(args):
    for db in tqdm(args.database, desc="Database converted."):
        trackedObjects = load_joblib_tracks(db)
        new_trackedOjects = trackedObjects_old_to_new(trackedObjects) 
        dump(new_trackedOjects, db, compress="lz4")

def diff(x1, x2, h):
    if h == 0:
        return 0
    return (x2-x1) / h

def dt(t1, t2):
    return t2-t1

def diffmap(a: np.array, t: np.array, k: int):
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

def submodule_function_5(args):
    if len(args.database) != len(args.output):
        print("Input and output arguments must be the same lenght.")
        raise ValueError()
    for db, out in tqdm(zip(args.database, args.output)):
        trackedObjects = load_joblib_tracks(db)
        trackedObjects_new = []
        for obj in trackedObjects:
            tmp = deepcopy(obj)
            T = np.array([d.frameID for d in tmp.history])
            tmp.history_VX_calculated, tmp.history_VT= diffmap(tmp.history_X, T, args.k_velocity)
            tmp.history_VY_calculated, _ = diffmap(tmp.history_Y, T, args.k_velocity)
            tmp.history_AX_calculated, _ = diffmap(tmp.history_VX_calculated, tmp.history_VT, args.k_accel)
            tmp.history_AY_calculated, _ = diffmap(tmp.history_VY_calculated, tmp.history_VT, args.k_accel)

            tmp.history_VX_calculated = np.insert(tmp.history_VX_calculated, 0, [0])
            tmp.history_VY_calculated = np.insert(tmp.history_VY_calculated, 0, [0])
            tmp.history_AX_calculated = np.insert(tmp.history_AX_calculated, 0, [0,0])
            tmp.history_AY_calculated = np.insert(tmp.history_AY_calculated, 0, [0,0])

            trackedObjects_new.append(tmp)
        dump(trackedObjects_new, out, compress="lz4")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-db", "--database", help="Path to database.", type=str, nargs='+')
    argparser.add_argument("--n_jobs", help="Paralell jobs to run.", type=int, default=16, required=True)
    argparser.set_defaults(func=mainmodule_function)

    subparser = argparser.add_subparsers(help="Submodules.")
    
    parser_training_dataset = subparser.add_parser("training", help="Extract the clustered track dataset with labels, for classifier training.")
    #argparser.add_argument("--training", help="Extract the filtered tracks with labels.", action="store_true", default=False)
    parser_training_dataset.add_argument("--min_samples", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--max_eps", help="Parameter for optics clustering", default=np.inf, type=float)
    parser_training_dataset.add_argument("--xi", help="Parameter for optics clustering", default=0.05, type=float)
    parser_training_dataset.add_argument("--min_cluster_size", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--threshold", help="Threshold for track filtering. The distance to the edges of the camera footage.", default=0.5, type=float)
    parser_training_dataset.add_argument("--cluster_dimensions", choices=["4D", "6D"], help="Choose feature vector type for clustering.", default="6D")
    parser_training_dataset.set_defaults(func=submodule_function)

    parser_mergeDatasets = subparser.add_parser("merge", help="Merge two or more joblib datasets.")
    parser_mergeDatasets.add_argument("--output", required=True, help="Output path and name of the file.")
    parser_mergeDatasets.set_defaults(func=submodule_function_2)

    downscaler_parser = subparser.add_parser("downscale", help="Normalize trackedObjects dataset.")
    downscaler_parser.add_argument("--video", help="Path to src video.")
    downscaler_parser.set_defaults(func=submodule_function_3)

    old_to_new_parser = subparser.add_parser("old2new", help="Update old TrackedObject dataset with history_X and history_Y fields.")
    old_to_new_parser.set_defaults(func=submodule_function_4)

    velacc_parser = subparser.add_parser("updateVelocityAccelarationVectors", help="Update old TrackedObject dataset with history_VX_calculated, history_VY_calculated, history_AX_calculated, history_AY_calculated fields.")
    velacc_parser.add_argument("--k_velocity", type=int, default=10, help="Stepsize for x, y derivation.")
    velacc_parser.add_argument("--k_accel", type=int, default=2, help="Stepsize for vx, vy derivation.")
    velacc_parser.add_argument("--output", nargs='+', type=str, help="Output path of joblib file.")
    velacc_parser.set_defaults(func=submodule_function_5)

    args = argparser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()