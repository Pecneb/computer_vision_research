from processing_utils import (
    tracks2joblib,
    trackslabels2joblib, 
    mergeDatasets,
    downscale_TrackedObjects,
    trackedObjects_old_to_new,
    load_dataset
)
import argparse
import cv2
from joblib import dump
import numpy as np
from tqdm import tqdm
from dataManagementClasses import TrackedObject
from copy import deepcopy
from itertools import starmap
from pathlib import Path

def mainmodule_function(args): 
    path = Path(args.database[0])
    if path.is_dir():
        for db in tqdm(path.glob("*.db"), desc="Database converted."):
            tracks2joblib(db, args.n_jobs)
    else:
        for db in tqdm(args.database, desc="Database converted."):
            tracks2joblib(db, args.n_jobs)

def submodule_function(args):
    trackslabels2joblib(args.database[0], args.output, args.min_samples, args.max_eps, args.xi, args.min_cluster_size , args.n_jobs, args.threshold, args.p_norm, args.cluster_dimensions)

def submodule_function_2(args):
    if len(args.database) < 2:
        return None
    databases = []
    for path in args.database:
        databases.append(load_dataset(path))
    merged = mergeDatasets(databases)
    dump(merged, args.output, compress="lz4")

def submodule_function_3(args):
    trackedObjects = load_dataset(args.database[0])
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
    for db in tqdm(args.database, desc="Database converter."):
        trackedObjects = load_dataset(db)
        new_trackedOjects = trackedObjects_old_to_new(trackedObjects) 
        dump(new_trackedOjects, db, compress="lz4")
    
def submodule_preprocess(args):
    from pathlib import Path
    from processing_utils import save_filtered_dataset
    datasetPath = Path(args.database[0])
    if datasetPath.is_dir():
        for ds in datasetPath.glob("*.joblib"):
            save_filtered_dataset(dataset=ds, 
                threshold=args.threshold, 
                max_dist=args.enter_exit_distance,
                euclidean_filtering=args.euclid,
                outdir=args.outdir)
    elif datasetPath.suffix==".joblib":
            save_filtered_dataset(dataset=datasetPath, 
                threshold=args.threshold, 
                max_dist=args.enter_exit_distance,
                euclidean_filtering=args.euclid,
                outdir=args.outdir)
    else:
        print("Non supported dataset format.")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-db", "--database", help="Path to database.", type=str, nargs='+')
    argparser.add_argument("--n-jobs", help="Paralell jobs to run.", type=int, default=16, required=True)
    argparser.set_defaults(func=mainmodule_function)

    subparser = argparser.add_subparsers(help="Submodules.")
    
    parser_training_dataset = subparser.add_parser("optics-cluster", help="Extract the clustered track dataset with labels, for classifier training.")
    #argparser.add_argument("--training", help="Extract the filtered tracks with labels.", action="store_true", default=False)
    parser_training_dataset.add_argument("--min-samples", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--max-eps", help="Parameter for optics clustering", default=np.inf, type=float)
    parser_training_dataset.add_argument("--xi", help="Parameter for optics clustering", default=0.05, type=float)
    parser_training_dataset.add_argument("--min-cluster_size", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--threshold", help="Threshold for track filtering. The distance to the edges of the camera footage.", default=0.5, type=float)
    parser_training_dataset.add_argument("--cluster-dimensions", choices=["4D", "6D"], help="Choose feature vector type for clustering.", default="6D")
    parser_training_dataset.add_argument("-p", "--p-norm", type=int, default=2, help="Set the p norm of the distance metrics.")
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

    preprocess_dataset_parser = subparser.add_parser("preprocess", help="Run preprocessing on dataset.")
    preprocess_dataset_parser.add_argument("--outdir", help="Output directory path.", required=True)
    preprocess_dataset_parser.add_argument("--threshold", type=float, default=0.7, help="Min-max filtering threshold value. Default: 0.7")
    preprocess_dataset_parser.add_argument("--enter-exit-distance", type=float, default=0.4, help="Minimum euclidean distance between enter and exit points. Default: 0.4")
    preprocess_dataset_parser.add_argument("--euclid", action="store_true", default=False, help="Set this flag to run euclidean"
        "distance based filtering on consecutive detections in trajectories. Default: False")
    preprocess_dataset_parser.add_argument("--det-distance", type=float, default=0.05, help="Maximum euclidean distance between"
        "consecutive detections in a trajectory. Default: 0.05")
    preprocess_dataset_parser.set_defaults(func=submodule_preprocess)

    args = argparser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()