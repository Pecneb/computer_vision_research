from processing_utils import random_split_tracks, load_joblib_tracks
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", help="Dataset")
    args = parser.parse_args()

    tracks = load_joblib_tracks(args.db) 

    print("\nFiltered tracks")
    random_split_tracks(tracks, 0.7, 1)
    print("\n")

if __name__ == "__main__":
    main()