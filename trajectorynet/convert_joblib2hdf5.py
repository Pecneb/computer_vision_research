import argparse

from utility.dataset import joblib2h5py


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a joblib file to a h5py file."
    )
    parser.add_argument("joblib_file", type=str, help="Path to joblib file.")
    parser.add_argument("h5py_file", type=str, help="Path to h5py file.")
    return parser.parse_args()

def main():
    args = get_args()
    joblib2h5py(args.joblib_file, args.h5py_file)


if __name__ == "__main__":
    main()
