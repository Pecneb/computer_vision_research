from processing_utils import random_split_tracks, load_joblib_tracks
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", help="Dataset")
    parser.add_argument("--seed", help="Integer as seed, to be able to regenerate the dataset.", type=int)
    args = parser.parse_args()

    tracks = load_joblib_tracks(args.db) 

    print("\nFiltered tracks")
    random_split_tracks(tracks, 0.7, args.seed)
    print("\n")

if __name__ == "__main__":
    main()