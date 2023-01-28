from processing_utils import tracks2joblib, trackslabels2joblib, mergeDatasets
import argparse

def mainmodule_function(args): 
    tracks2joblib(args.database[0], args.n_jobs)

def submodule_function(args):
    trackslabels2joblib(args.database[0], args.min_samples, args.max_eps, args.xi, args.min_cluster_size , args.n_jobs, args.threshold)

def submodule_function_2(args):
    mergeDatasets(args.database, args.output)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-db", "--database", help="Path to database.", type=str, nargs='+')
    argparser.add_argument("--n_jobs", help="Paralell jobs to run.", type=int, default=16)
    argparser.set_defaults(func=mainmodule_function)

    subparser = argparser.add_subparsers(help="Submodules.")
    
    parser_training_dataset = subparser.add_parser("training", help="Extract the clustered track dataset with labels, for classifier training.")
    #argparser.add_argument("--training", help="Extract the filtered tracks with labels.", action="store_true", default=False)
    parser_training_dataset.add_argument("--min_samples", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--max_eps", help="Parameter for optics clustering", default=0.2, type=float)
    parser_training_dataset.add_argument("--xi", help="Parameter for optics clustering", default=0.15, type=float)
    parser_training_dataset.add_argument("--min_cluster_size", help="Parameter for optics clustering", default=10, type=int)
    parser_training_dataset.add_argument("--threshold", help="Threshold for track filtering. The distance to the edges of the camera footage.", default=0.5, type=float)
    parser_training_dataset.set_defaults(func=submodule_function)

    parser_mergeDatasets = subparser.add_parser("merge", help="Merge two or more joblib datasets.")
    parser_mergeDatasets.add_argument("--output", required=True, help="Output path and name of the file.")
    parser_mergeDatasets.set_defaults(func=submodule_function_2)

    args = argparser.parse_args()

    args.func(args)

if __name__ == '__main__':
    main()