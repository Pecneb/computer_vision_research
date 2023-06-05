import argparse
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS
from processing_utils import (
    load_dataset, 
    filter_trajectories, 
    make_4D_feature_vectors
)
from clustering import clustering_on_feature_vectors

def trafficHistogram(args):
    tracks = load_dataset(args.database)
    if not args.filtered:
        tracks_filtered = filter_trajectories(tracks, 0.7)
        X_ = make_4D_feature_vectors(tracks_filtered)
    else:
        X_ = make_4D_feature_vectors(tracks)
    _, labels = clustering_on_feature_vectors(
        X_, OPTICS, 
        min_samples=args.min_samples,
        max_eps=args.max_eps,
        xi=args.xi,
        p=args.p_norm
    )
    Y = labels[labels > -1]
    X = X_[labels > -1]
    fig, ax = plt.subplots(1,1,figsize=(15,15))
    classes = list(set(Y))
    #TODO arrange X into matrix, with corresponding rows for each class
    ax.hist(X, classes)
    plt.show()
    plt.close()

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
    clusterHistogram.set_defaults(func=trafficHistogram)
    args = argparser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()