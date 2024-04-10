### System Imports ###
import argparse
from pathlib import Path
from typing import List


### Third Party Imports ###
import pandas as pd


def get_args():
    """
    Get command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot the features structure")
    parser.add_argument(
        "-r",
        "--evaluation-results",
        type=str,
        required=True,
        help="Path to the evaluation results",
        nargs="+",
    )
    return parser.parse_args()


def load_data(paths: List[str]) -> pd.DataFrame:
    """Load multiple csv files and concatenate them into a single DataFrame

    Parameters
    ----------
    paths : List[str]
        List of paths to the csv files

    Returns
    -------
    pd.DataFrame
        Dataframe containing the concatenated data
    """
    dataframes = []
    for path in paths:
        if not Path(path).exists():
            print(f"Path {path} does not exist")
        else:
            dataframes.append(pd.read_csv(path))
    return pd.concat(dataframes)


def main():
    args = get_args()
    df = load_data(args.evaluation_results) 
    mlp_results = df.where(df["classifier"].str.contains("MLP")).dropna()
    svm_results = df.where(df["classifier"].str.contains("SVM")).dropna()
    mlp_results["classifier"] = "MLP"
    svm_results["classifier"] = "SVM"
    print(mlp_results.sort_values("version"))
    print(svm_results.sort_values("version"))


if __name__ == "__main__":
    main()
