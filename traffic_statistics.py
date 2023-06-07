import argparse
import logging
import time
import os
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS
from typing import List
from processing_utils import (
    load_dataset, 
    filter_trajectories, 
    make_4D_feature_vectors,
    calc_cluster_centers,
    upscale_cluster_centers
)
from clustering import clustering_on_feature_vectors
from dataManagementClasses import TrackedObject

logging.basicConfig(level=logging.INFO)

def trafficHistogram(dataset: List[TrackedObject], output: str, bg_img: str, **estkwargs):
    """Plot histogram about traffic flow.

    Args:
        dataset (List[TrackedObject]): Trajectory dataset. 
        output (str): Output directory path, where plots will be saved.
                      If output is None, plots will not be saved. 
        bg_img (str): Background image of the heatmap like plot.
    """
    from matplotlib import colors
    logging.info(estkwargs)
    if output is not None:
        outpath = Path(output)
        if not outpath.exists():
            outpath.mkdir(parents=True)
    X = make_4D_feature_vectors(dataset)
    _, labels = clustering_on_feature_vectors(
        X, OPTICS, 
        **estkwargs
    )
    Y = labels[labels > -1]
    X = dataset[labels > -1]
    enter_cluster_center = calc_cluster_centers(X, Y, False)
    exit_cluster_center = calc_cluster_centers(X, Y, True)
    fig1, ax1 = plt.subplots(1,1,figsize=(15,15))
    ax1.set_title(f"{output} clusters histogram")
    classes = np.array(list(set(Y)))
    N, bins, patches = ax1.hist(Y, classes-0.5, align="mid", range=(classes.min(), classes.max()), edgecolor="black")
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        #TODO more vivid colors
        color = plt.cm.jet(norm(thisfrac)) # jet colors seem to be nice
        thispatch.set_facecolor(color)
    fig1.colorbar(plt.cm.ScalarMappable(norm, plt.cm.jet), ax=ax1, location='right')
    if output is not None:
        fig1Name = os.path.join(output, "histogram.png")
        fig1.savefig(fig1Name)
        logging.info(f"Fig1: \"{fig1Name}\"")
    if bg_img is not None:
        fig2, ax2 = plt.subplots(1,1,figsize=(15,15))
        ax2.set_title(f"{output} cluster heatmap")
        bgImg = plt.imread(bg_img)
        mp = ax2.imshow(bgImg)
        upscaled_enters = upscale_cluster_centers(enter_cluster_center, bgImg.shape[1], bgImg.shape[0])
        upscaled_exits = upscale_cluster_centers(exit_cluster_center, bgImg.shape[1], bgImg.shape[0])
        for p, q, thisfrac in zip(upscaled_enters, upscaled_exits, fracs):
            color = plt.cm.jet(norm(thisfrac)) # jet colors seem to be nice
            xx = np.vstack((p[0], q[0]))
            yy = np.vstack((p[1], q[1]))
            ax2.plot(xx, yy, color=color, marker='o', linestyle='-')
        fig2.colorbar(plt.cm.ScalarMappable(norm, plt.cm.jet), ax=ax2, location='bottom')
        if output is not None:
            fig2Name = os.path.join(output, "clusters.png")
            fig2.savefig(fig2Name)
            logging.info(f"Fig2: \"{fig2Name}\"")
    plt.show()
    plt.close()
    return Y 

def extractHourlyData(dataset: List[TrackedObject]):
    fps = 30
    fph = fps*60*60
    maxFrameNum = dataset[-1].history[0].frameID
    hourNum = int(np.ceil([maxFrameNum/fph])[0])
    hourlyData = np.zeros(shape=(hourNum, len(dataset)), dtype=TrackedObject)
    counter = np.zeros(shape=(hourNum), dtype=int)
    for i in range(len(dataset)):
        print(dataset[i].history[0].frameID)
        actHour = int(dataset[i].history[0].frameID//fph)
        hourlyData[actHour, counter[actHour]] = dataset[i]
    return hourlyData

def trafficHistogramModule(args):
    logging.info("Traffic histogram module started")
    start = time.time()
    tracks = load_dataset(args.database[0])
    if not args.filtered:
        tracks_filtered = filter_trajectories(tracks, 0.7)
    else:
        tracks_filtered = tracks
    trafficHistogram(tracks_filtered, 
                     args.output,
                     args.bg_img,
                     min_samples=args.min_samples,
                     max_eps=args.max_eps,
                     xi=args.xi,
                     p=args.p_norm)
    logging.info(f"Traffic histogram module ran for {time.time()-start} seconds")

def hourlyStatisticsModule(args):
    logging.info("Traffic hourly statistics module started")
    start = time.time()
    tracksHourly = [] 
    for ds in args.database:
        tmpTracks = load_dataset(ds)
        if not args.filtered:
            tracksHourly.append(filter_trajectories(tmpTracks, 0.7))
        else:
            tracksHourly.append(tmpTracks)
        extractHourlyData(tracksHourly[-1])
    logging.info(f"Traffic hourly statistics module ran for {time.time()-start} seconds")

def trafficStatisticTable(args):
    import pandas as pd
    logging.info("Traffic histogram module started")
    start = time.time()
    tracks = load_dataset(args.database[0])
    if not args.filtered:
        tracks_filtered = filter_trajectories(tracks, 0.7)
    else:
        tracks_filtered = tracks
    trafficHistogram(tracks_filtered, 
                     args.output,
                     args.bg_img,
                     min_samples=args.min_samples,
                     max_eps=args.max_eps,
                     xi=args.xi,
                     p=args.p_norm)
    logging.info(f"Traffic histogram module ran for {time.time()-start} seconds")
    #TODO statistical table about clusters
    df = pd.DataFrame()

def main():
    argparser = argparse.ArgumentParser("Traffic statistics plotter")
    argparser.add_argument("-db", "--database", nargs='+', help="Database path.")
    argparser.add_argument("-o", "--output", help="Output directory path of the plots.")
    argparser.add_argument("--filtered", action="store_true", default=False, help="If dataset is already filtered use this flag.")
    subParser = argparser.add_subparsers(help="Plot statistical data about the traffic data collected in runtime.")

    clusterHistogram = subParser.add_parser("histogram", help="Plot histogram of clusters.")
    clusterHistogram.add_argument("--min_samples", default=20, type=int, help="OPTICS min samples param.")
    clusterHistogram.add_argument("--max_eps", type=float, default=0.2, help="OPTICS max epsilon distance param.")
    clusterHistogram.add_argument("--xi", type=float, default=0.15, help="OPTICS xi param.")
    clusterHistogram.add_argument("--p_norm", type=int, default=2, help="OPTICS p norm param.")
    clusterHistogram.add_argument("--bg_img", help="Background image path.")
    clusterHistogram.set_defaults(func=trafficHistogramModule)

    hourlyClusterStats = subParser.add_parser("hourly", help="Plot hourly statistic.")
    hourlyClusterStats.add_argument("--min_samples", default=20, type=int, help="OPTICS min samples param.")
    hourlyClusterStats.add_argument("--max_eps", type=float, default=0.2, help="OPTICS max epsilon distance param.")
    hourlyClusterStats.add_argument("--xi", type=float, default=0.15, help="OPTICS xi param.")
    hourlyClusterStats.add_argument("--p_norm", type=int, default=2, help="OPTICS p norm param.")
    hourlyClusterStats.add_argument("--bg_img", help="Background image path.")
    hourlyClusterStats.set_defaults(func=hourlyStatisticsModule)

    args = argparser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()