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
    trackslabels2joblib(args.database[0], args.output, args.min_samples, args.max_eps, args.xi, args.min_cluster_size , args.n_jobs, args.threshold, args.p_norm, args.cluster_dimensions)

def submodule_function_2(args):
    mergeDatasets(args.database, args.output)

def submodule_function_3(args):
    trackedObjects = load_joblib_tracks(args.database[0])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError()
    ret, img = cap.read()
    if ret:
        downscaled = downscale_TrackedObjects(trackedObjects, img)
        dump(downscaled, args.database[0], compress="lz4")
    else:
        raise IOError()

def submodule_function_4(args):
    for db in tqdm(args.database, desc="Database converted."):
        trackedObjects = load_joblib_tracks(db)
        new_trackedOjects = trackedObjects_old_to_new(trackedObjects) 
        dump(new_trackedOjects, db, compress="lz4")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-db", "--database", help="Path to database.", type=str, nargs='+')
    argparser.add_argument("--n_jobs", help="Paralell jobs to run.", type=int, default=16, required=True)
    argparser.set_defaults(func=mainmodule_function)

    subparser = argparser.add_subparsers(help="Submodules.")
    
    parser_training_dataset = subparser.add_parser("optics-cluster", help="Extract the clustered track dataset with labels, for classifier training.")
    #argparser.add_argument("--training", help="Extract the filtered tracks with labels.", action="store_true", default=False)
    parser_training_dataset.add_argument("--min_samples", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--max_eps", help="Parameter for optics clustering", default=np.inf, type=float)
    parser_training_dataset.add_argument("--xi", help="Parameter for optics clustering", default=0.05, type=float)
    parser_training_dataset.add_argument("--min_cluster_size", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--threshold", help="Threshold for track filtering. The distance to the edges of the camera footage.", default=0.5, type=float)
    parser_training_dataset.add_argument("--cluster_dimensions", choices=["4D", "6D"], help="Choose feature vector type for clustering.", default="6D")
    parser_training_dataset.add_argument("-p", "--p_norm", type=int, default=2, help="Set the p norm of the distance metrics.")
    parser_training_dataset.add_argument("--output", type=str, help="Output path.")
    parser_training_dataset.set_defaults(func=submodule_function)

    parser_mergeDatasets = subparser.add_parser("merge", help="Merge two or more joblib datasets.")
    parser_mergeDatasets.add_argument("--output", required=True, help="Output path and name of the file.")
    parser_mergeDatasets.set_defaults(func=submodule_function_2)

    downscaler_parser = subparser.add_parser("downscale", help="Normalize trackedObjects dataset.")
    downscaler_parser.add_argument("--video", help="Path to src video.")
    downscaler_parser.set_defaults(func=submodule_function_3)

    old_to_new_parser = subparser.add_parser("old2new", help="Update old TrackedObject dataset with history_X and history_Y fields.")
    old_to_new_parser.set_defaults(func=submodule_function_4)

    args = argparser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()