from processing_utils import tracks2joblib, trackslabels2joblib
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-db", "--database", help="Path to database.", type=str)
    argparser.add_argument("--n_jobs", help="Paralell jobs to run.", type=int, default=16)
    argparser.add_argument("--filtered", help="Extract the filtered tracks with labels.", action="store_true", default=False)
    argparser.add_argument("--min_samples", help="Parameter for optics clustering", default=10, type=int)
    argparser.add_argument("--max_eps", help="Parameter for optics clustering", default=0.2, type=float)
    argparser.add_argument("--xi", help="Parameter for optics clustering", default=0.15, type=float)
    argparser.add_argument("--min_cluster_size", help="Parameter for optics clustering", default=10)
    args = argparser.parse_args()
    if args.database is None:
        argparser.print_help()

    if args.filtered: 
        trackslabels2joblib(args.database, args.min_samples, args.max_eps, args.xi, args.min_cluster_size, args.n_jobs)
    else:
        tracks2joblib(args.database, args.n_jobs)

if __name__ == '__main__':
    main()