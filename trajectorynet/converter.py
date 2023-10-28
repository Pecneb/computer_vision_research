### System ###
### Third Pary ###
import argparse
import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from clustering import make_4D_feature_vectors, make_6D_feature_vectors
from dataManagementClasses import TrackedObject
from joblib import dump
from tqdm import tqdm
### Local ###
from utility.dataset import (downscale_TrackedObjects, load_dataset,
                             mergeDatasets,
                             preprocess_database_data_multiprocessed,
                             tracks2joblib)
from utility.general import diffmap
from utility.preprocessing import filter_by_class, filter_trajectories

logging.basicConfig(filename="processing_utils.log", level=logging.DEBUG)


def save_filtered_dataset(dataset: Union[str, Path], threshold: float, max_dist: float, euclidean_filtering: bool = False, outdir=None):
    """Filter dataset and save to joblib binary.

    Args:
        dataset (str): Dataset path 
        threshold (float): Filter threshold 
    """
    if "filtered" in str(dataset):
        print(f"{dataset} is already filtered.")
        return False
    try:
        trajectories = load_dataset(dataset)
    except EOFError as e:
        print(e)
        return False
    logging.info(f"Trajectories \"{dataset}\" loaded")
    trajs2save = filter_trajectories(
        trackedObjects=trajectories,
        threshold=threshold,
        detectionDistanceFiltering=euclidean_filtering,
        detDist=max_dist
    )
    logging.info(
        f"Dataset filtered. Trajectories reduced from {len(trajectories)} to {len(trajs2save)}")
    datasetPath = Path(dataset)
    if outdir is not None:
        outPath = Path(outdir)
        if not outPath.exists():
            outPath.mkdir()
        filteredDatasetPath = outPath.joinpath(
            datasetPath.stem+"_filtered.joblib")
    else:
        filteredDatasetPath = datasetPath.parent.joinpath(
            datasetPath.stem+"_filtered.joblib")
    dump(trajs2save, filteredDatasetPath)
    logging.info(f"Filtered dataset saved to {filteredDatasetPath.absolute()}")


def trackslabels2joblib(path2tracks: str, output: str, min_samples=10, max_eps=0.2, xi=0.15, min_cluster_size=10, n_jobs=18, threshold=0.5, p=2, cluster_dimensions: str = "4D"):
    """Save training tracks with class numbers ordered to them.

    Args:
        path2tracks (str): Path to dataset. 
        min_samples (int, optional): Optics clustering parameter. Defaults to 10.
        max_eps (float, optional): Optics clustering parameter. Defaults to 0.2.
        xi (float, optional): Optics clustering parameter. Defaults to 0.15.
        min_cluster_size (int, optional): Optics clustering parameter. Defaults to 10.
        n_jobs (int, optional): Number of processes to run. Defaults to 18.

    Returns:
        _type_: _description_
    """
    from clustering import clustering_on_feature_vectors
    from sklearn.cluster import OPTICS
    filext = path2tracks.split('/')[-1].split('.')[-1]

    if filext == 'db':
        tracks = preprocess_database_data_multiprocessed(path2tracks, n_jobs)
    elif filext == 'joblib':
        tracks = load_dataset(path2tracks)
    else:
        print("Error: Wrong file type.")
        return False

    tracks_filtered = filter_trajectories(tracks, threshold=threshold)
    tracks_car_only = filter_by_class(tracks_filtered)

    if cluster_dimensions == "6D":
        cluster_features = make_6D_feature_vectors(tracks_car_only)
    else:
        cluster_features = make_4D_feature_vectors(tracks_car_only)
    _, labels = clustering_on_feature_vectors(cluster_features, OPTICS,
                                              n_jobs=n_jobs,
                                              p=p,
                                              min_samples=min_samples,
                                              max_eps=max_eps,
                                              xi=xi,
                                              min_cluster_size=min_cluster_size)

    # order labels to tracks, store it in a list[dictionary] format
    tracks_classes = []
    for i, t in enumerate(tracks_car_only):
        tracks_classes.append({
            "track": t,
            "class": labels[i]
        })

    # filename = path2tracks.split('/')[-1].split('.')[0] + '_clustered.joblib'
    # avepath = os.path.join("research_data", path2tracks.split('/')[-1].split('.')[0], filename)

    print("Saving: ", output)

    return dump(tracks_classes, output, compress="lz4")


def trackedObjects_old_to_new(trackedObjects: list, k_velocity: int = 10, k_accel: int = 2):
    """Depracated function. Archived.

    Args:
        trackedObjects (list[TrackedObject]): tracked objects. 

    Returns:
        list: list of converted tracked objects 
    """
    new_trackedObjects = []
    for t in tqdm.tqdm(trackedObjects, desc="TrackedObjects converted to new class structure."):
        tmp_obj = TrackedObject(t.objID, t.history[0])
        tmp_obj.history = t.history
        tmp_obj.history_X = np.array([d.X for d in t.history])
        tmp_obj.history_Y = np.array([d.Y for d in t.history])
        tmp_obj.X = t.X
        tmp_obj.Y = t.Y
        tmp_obj.VX = t.VX
        tmp_obj.VY = t.VY
        tmp_obj.AX = t.AX
        tmp_obj.AY = t.AY
        tmp_obj.futureX = t.futureX
        tmp_obj.futureY = t.futureY
        tmp_obj.isMoving = t.isMoving
        tmp_obj.label = t.label
        T = np.array([d.frameID for d in tmp_obj.history])
        tmp_obj.history_VX_calculated, tmp_obj.history_VT = diffmap(
            tmp_obj.history_X, T, k_velocity)
        tmp_obj.history_VY_calculated, _ = diffmap(
            tmp_obj.history_Y, T, k_velocity)
        tmp_obj.history_AX_calculated, _ = diffmap(
            tmp_obj.history_VX_calculated, tmp_obj.history_VT, k_accel)
        tmp_obj.history_AY_calculated, _ = diffmap(
            tmp_obj.history_VY_calculated, tmp_obj.history_VT, k_accel)
        tmp_obj.history_VX_calculated = np.insert(
            tmp_obj.history_VX_calculated, 0, [0])
        tmp_obj.history_VY_calculated = np.insert(
            tmp_obj.history_VY_calculated, 0, [0])
        tmp_obj.history_AX_calculated = np.insert(
            tmp_obj.history_AX_calculated, 0, [0, 0])
        tmp_obj.history_AY_calculated = np.insert(
            tmp_obj.history_AY_calculated, 0, [0, 0])
        new_trackedObjects.append(tmp_obj)
    return new_trackedObjects


def mainmodule_function(args):
    path = Path(args.database[0])
    if path.is_dir():
        for db in tqdm(path.glob("*.db"), desc="Database converted."):
            tracks2joblib(db, args.n_jobs)
    else:
        for db in tqdm(args.database, desc="Database converted."):
            tracks2joblib(db, args.n_jobs)


def submodule_function(args):
    trackslabels2joblib(args.database[0], args.output, args.min_samples, args.max_eps, args.xi,
                        args.min_cluster_size, args.n_jobs, args.threshold, args.p_norm, args.cluster_dimensions)


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
    datasetPath = Path(args.database[0])
    if datasetPath.is_dir():
        for ds in datasetPath.glob("*.joblib"):
            print(ds)
            save_filtered_dataset(dataset=ds,
                                  threshold=args.threshold,
                                  max_dist=args.enter_exit_distance,
                                  euclidean_filtering=args.euclid,
                                  outdir=args.outdir)
    elif datasetPath.suffix == ".joblib":
        save_filtered_dataset(dataset=datasetPath,
                              threshold=args.threshold,
                              max_dist=args.enter_exit_distance,
                              euclidean_filtering=args.euclid,
                              outdir=args.outdir)
    else:
        print("Non supported dataset format.")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-db", "--database",
                           help="Path to database.", type=str, nargs='+')
    argparser.add_argument(
        "--n-jobs", help="Paralell jobs to run.", type=int, default=16, required=True)
    argparser.set_defaults(func=mainmodule_function)

    subparser = argparser.add_subparsers(help="Submodules.")

    parser_training_dataset = subparser.add_parser(
        "optics-cluster", help="Extract the clustered track dataset with labels, for classifier training.")
    # argparser.add_argument("--training", help="Extract the filtered tracks with labels.", action="store_true", default=False)
    parser_training_dataset.add_argument(
        "--min-samples", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument(
        "--max-eps", help="Parameter for optics clustering", default=np.inf, type=float)
    parser_training_dataset.add_argument(
        "--xi", help="Parameter for optics clustering", default=0.05, type=float)
    parser_training_dataset.add_argument(
        "--min-cluster_size", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument(
        "--threshold", help="Threshold for track filtering. The distance to the edges of the camera footage.", default=0.5, type=float)
    parser_training_dataset.add_argument("--cluster-dimensions", choices=[
                                         "4D", "6D"], help="Choose feature vector type for clustering.", default="6D")
    parser_training_dataset.add_argument(
        "-p", "--p-norm", type=int, default=2, help="Set the p norm of the distance metrics.")
    parser_training_dataset.add_argument(
        "--output", type=str, help="Output path.")
    parser_training_dataset.set_defaults(func=submodule_function)

    parser_mergeDatasets = subparser.add_parser(
        "merge", help="Merge two or more joblib datasets.")
    parser_mergeDatasets.add_argument(
        "--output", required=True, help="Output path and name of the file.")
    parser_mergeDatasets.set_defaults(func=submodule_function_2)

    downscaler_parser = subparser.add_parser(
        "downscale", help="Normalize trackedObjects dataset.")
    downscaler_parser.add_argument("--video", help="Path to src video.")
    downscaler_parser.set_defaults(func=submodule_function_3)

    old_to_new_parser = subparser.add_parser(
        "old2new", help="Update old TrackedObject dataset with history_X and history_Y fields.")
    old_to_new_parser.set_defaults(func=submodule_function_4)

    preprocess_dataset_parser = subparser.add_parser(
        "preprocess", help="Run preprocessing on dataset.")
    preprocess_dataset_parser.add_argument(
        "--outdir", help="Output directory path.", required=True)
    preprocess_dataset_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Min-max filtering threshold value. Default: 0.7")
    preprocess_dataset_parser.add_argument("--enter-exit-distance", type=float, default=0.4,
                                           help="Minimum euclidean distance between enter and exit points. Default: 0.4")
    preprocess_dataset_parser.add_argument("--euclid", action="store_true", default=False, help="Set this flag to run euclidean"
                                           "distance based filtering on consecutive detections in trajectories. Default: False")
    preprocess_dataset_parser.add_argument("--det-distance", type=float, default=0.05, help="Maximum euclidean distance between"
                                           "consecutive detections in a trajectory. Default: 0.05")
    preprocess_dataset_parser.set_defaults(func=submodule_preprocess)

    args = argparser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
