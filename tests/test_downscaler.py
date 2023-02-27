from dataManagementClasses import TrackedObject
from dataManagementClasses import Detection
from processing_utils import load_joblib_tracks
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("dataset") 
    args = argparser.parse_args()

    tracks = load_joblib_tracks(args.dataset)
    for i in range(int(len(tracks)*0.1)):
        print(TrackedObject.downscale_feature(tracks[i].feature_(), 1920, 1080))

if __name__ == "__main__":
    main()