import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS
from processing_utils import (
    load_dataset, 
    filter_trajectories, 
    make_4D_feature_vectors,
    calc_cluster_centers,
    upscale_cluster_centers
)
from clustering import clustering_on_feature_vectors
from dataManagementClasses import TrackedObject

def trafficHistogram(args):
    from matplotlib import colors
    tracks = load_dataset(args.database)
    if not args.filtered:
        tracks_filtered = filter_trajectories(tracks, 0.7)
        X_ = make_4D_feature_vectors(tracks_filtered)
    else:
        tracks_filtered = tracks
        X_ = make_4D_feature_vectors(tracks)
    _, labels = clustering_on_feature_vectors(
        X_, OPTICS, 
        min_samples=args.min_samples,
        max_eps=args.max_eps,
        xi=args.xi,
        p=args.p_norm
    )
    Y = labels[labels > -1]
    X = tracks_filtered[labels > -1]
    enter_cluster_center = calc_cluster_centers(X, Y, False)
    exit_cluster_center = calc_cluster_centers(X, Y, True)
    fig, ax = plt.subplots(nrows=2, figsize=(15,15))
    classes = np.array(list(set(Y)))
    N, bins, patches = ax[0].hist(labels, classes-0.5, align="mid", range=(classes.min(), classes.max()), edgecolor="black")
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    print(N, fracs)
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    if args.bg_img is not None:
        bgImg = plt.imread(args.bg_img)
        ax[1].imshow(bgImg)
        upscaled_enters = upscale_cluster_centers(enter_cluster_center, bgImg.shape[1], bgImg.shape[0])
        upscaled_exits = upscale_cluster_centers(exit_cluster_center, bgImg.shape[1], bgImg.shape[0])
        for p, q, thisfrac in zip(upscaled_enters, upscaled_exits, fracs):
            color = plt.cm.viridis(norm(thisfrac))
            xx = np.vstack((p[0], q[0]))
            yy = np.vstack((p[1], q[1]))
            ax[1].plot(xx, yy, color=color, marker='o', linestyle='-')
    plt.show()
    plt.close()

def hourlyStatistics(args):
    #TODO
    pass

def trafficStatisticTable(args):
    #TODO
    pass

def main():
    argparser = argparse.ArgumentParser("Traffic statistics plotter")
    argparser.add_argument("-db", "--database", help="Database path.")
    argparser.add_argument("-o", "--output", help="Output directory path of the plots.")
    argparser.add_argument("--filtered", action="store_true", default=False, help="If dataset is already filtered use this flag.")
    subParser = argparser.add_subparsers(help="Plot statistical data about the traffic data collected in runtime.")
    clusterHistogram = subParser.add_parser("histogram", help="Plot histogram of clusters.")
    clusterHistogram.add_argument("--min_samples", default=20, type=int, help="OPTICS min samples param.")
    clusterHistogram.add_argument("--max_eps", type=float, default=0.2, help="OPTICS max epsilon distance param.")
    clusterHistogram.add_argument("--xi", type=float, default=0.15, help="OPTICS xi param.")
    clusterHistogram.add_argument("--p_norm", type=int, default=2, help="OPTICS p norm param.")
    clusterHistogram.add_argument("--bg_img", help="Background image path.")
    clusterHistogram.set_defaults(func=trafficHistogram)
    args = argparser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()