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
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to the output file",
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


def markdown_layout(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    return f"""
    MLP  
    KNN  

          Re                    ReVe                    ReVeAe
    {df1.where(df1["version"] == "Re").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "Re").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}         {df1.where(df1["version"] == "ReVe").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVe").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeAe").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeAe").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}
    {df2.where(df2["version"] == "Re").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "Re").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}         {df2.where(df2["version"] == "ReVe").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVe").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeAe").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeAe").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}

                                ReRs                    ReVeRs                   ReVeAeRs
                          {df1.where(df1["version"] == "ReRs").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReRs").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeRs").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeRs").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeAeRs").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeAeRs").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}
                          {df2.where(df2["version"] == "ReRs").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReRs").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeRs").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeRs").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeAeRs").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeAeRs").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}

    
                                ReRm                    ReVeRm                   ReVeAeRm
                          {df1.where(df1["version"] == "ReRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeAeRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeAeRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}
                          {df2.where(df2["version"] == "ReRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeAeRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeAeRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}
    
                                                        ReRsRm                   ReVeRsRm                 ReVeAeRsRm
                                                    {df1.where(df1["version"] == "ReRsRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReRsRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeRsRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeRsRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df1.where(df1["version"] == "ReVeAeRsRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df1.where(df1["version"] == "ReVeAeRsRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}
                                                    {df2.where(df2["version"] == "ReRsRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReRsRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeRsRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeRsRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}             {df2.where(df2["version"] == "ReVeAeRsRm").dropna()["balanced_test_score (percent)"].values[0]:.2f} | {df2.where(df2["version"] == "ReVeAeRsRm").dropna()["balanced_pooled_test_score (percent)"].values[0]:.2f}
    """


def main():
    args = get_args()
    df = load_data(args.evaluation_results)
    mlp_results = df.where(df["classifier"].str.contains("MLP")).dropna()
    knn_results = df.where(df["classifier"].str.contains("KNN")).dropna()
    mlp_results["classifier"] = "MLP"
    knn_results["classifier"] = "KNN"
    print(markdown_layout(mlp_results, knn_results))
    with open(args.output, "w") as f:
        f.write(markdown_layout(mlp_results, knn_results))


if __name__ == "__main__":
    main()
