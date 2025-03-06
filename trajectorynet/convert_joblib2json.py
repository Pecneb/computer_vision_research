import argparse
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
import json
import joblib

# Assuming the functions are defined in dataset.py
from utility.dataset import load_dataset, dataset_to_json, load_dataset_from_json, convert_joblib_to_json_chunked


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a joblib file to a JSON file."
    )
    parser.add_argument("joblib_file", type=str, help="Path to joblib file.")
    parser.add_argument("json_file", type=str, help="Path to JSON file.")
    parser.add_argument("--chunks", action="store_true", help="Use chunking to.")
    return parser.parse_args()


def load_joblib_file(path: Path) -> Optional[np.ndarray]:
    """Load a dataset from a joblib file.

    Parameters
    ----------
    path : Path
        Path to the joblib file.

    Returns
    -------
    np.ndarray
        Loaded dataset.
    """
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading joblib file: {e}")
        return None


def compare_datasets(original: np.ndarray, loaded: np.ndarray) -> bool:
    """Compare two datasets to check if they are identical.

    Parameters
    ----------
    original : np.ndarray
        Original dataset.
    loaded : np.ndarray
        Loaded dataset from JSON.

    Returns
    -------
    bool
        True if datasets are identical, False otherwise.
    """
    if len(original) != len(loaded):
        print(f"Dataset length mismatch: {len(original)} != {len(loaded)}")
        return False

    for idx, (orig_obj, loaded_obj) in enumerate(zip(original, loaded)):
        if orig_obj.objID != loaded_obj.objID:
            print(
                f"Object ID mismatch at index {idx}: {orig_obj.objID} != {loaded_obj.objID}"
            )
            return False
        if not np.array_equal(orig_obj.history_X, loaded_obj.history_X):
            print(f"history_X mismatch at index {idx}")
            return False
        if not np.array_equal(orig_obj.history_Y, loaded_obj.history_Y):
            print(f"history_Y mismatch at index {idx}")
            return False
        if not np.array_equal(
            orig_obj.history_VX_calculated, loaded_obj.history_VX_calculated
        ):
            print(f"history_VX_calculated mismatch at index {idx}")
            return False
        if not np.array_equal(
            orig_obj.history_VY_calculated, loaded_obj.history_VY_calculated
        ):
            print(f"history_VY_calculated mismatch at index {idx}")
            return False
        if not np.array_equal(
            orig_obj.history_AX_calculated, loaded_obj.history_AX_calculated
        ):
            print(f"history_AX_calculated mismatch at index {idx}")
            return False
        if not np.array_equal(
            orig_obj.history_AY_calculated, loaded_obj.history_AY_calculated
        ):
            print(f"history_AY_calculated mismatch at index {idx}")
            return False
        if len(orig_obj.history) != len(loaded_obj.history):
            print(
                f"History length mismatch at index {idx}: {len(orig_obj.history)} != {len(loaded_obj.history)}"
            )
            print(f"Original history: {orig_obj.history}")
            print(f"Loaded history: {loaded_obj.history}")
            print(f"Original history length: {len(orig_obj.history)}")
            print(f"Loaded history length: {len(loaded_obj.history)}")
            return False
        for det_idx, (orig_det, loaded_det) in enumerate(
            zip(orig_obj.history, loaded_obj.history)
        ):
            if orig_det.label != loaded_det.label:
                print(
                    f"Detection label mismatch at index {idx}, detection {det_idx}: {orig_det.label} != {loaded_det.label}"
                )
                return False
            if orig_det.confidence != loaded_det.confidence:
                print(
                    f"Detection confidence mismatch at index {idx}, detection {det_idx}: {orig_det.confidence} != {loaded_det.confidence}"
                )
                return False
            if orig_det.X != loaded_det.X:
                print(
                    f"Detection X mismatch at index {idx}, detection {det_idx}: {orig_det.X} != {loaded_det.X}"
                )
                return False
            if orig_det.Y != loaded_det.Y:
                print(
                    f"Detection Y mismatch at index {idx}, detection {det_idx}: {orig_det.Y} != {loaded_det.Y}"
                )
                return False
            if orig_det.Width != loaded_det.Width:
                print(
                    f"Detection Width mismatch at index {idx}, detection {det_idx}: {orig_det.Width} != {loaded_det.Width}"
                )
                return False
            if orig_det.Height != loaded_det.Height:
                print(
                    f"Detection Height mismatch at index {idx}, detection {det_idx}: {orig_det.Height} != {loaded_det.Height}"
                )
                return False
            if orig_det.frameID != loaded_det.frameID:
                print(
                    f"Detection frameID mismatch at index {idx}, detection {det_idx}: {orig_det.frameID} != {loaded_det.frameID}"
                )
                return False
            if orig_det.VX != loaded_det.VX:
                print(
                    f"Detection VX mismatch at index {idx}, detection {det_idx}: {orig_det.VX} != {loaded_det.VX}"
                )
                return False
            if orig_det.VY != loaded_det.VY:
                print(
                    f"Detection VY mismatch at index {idx}, detection {det_idx}: {orig_det.VY} != {loaded_det.VY}"
                )
                return False
            if orig_det.AX != loaded_det.AX:
                print(
                    f"Detection AX mismatch at index {idx}, detection {det_idx}: {orig_det.AX} != {loaded_det.AX}"
                )
                return False
            if orig_det.AY != loaded_det.AY:
                print(
                    f"Detection AY mismatch at index {idx}, detection {det_idx}: {orig_det.AY} != {loaded_det.AY}"
                )
                return False

    return True


def main():
    args = get_args()
    # Path to the original dataset
    original_dataset_path = args.joblib_file

    # Path to the JSON file
    json_file_path = args.json_file

    # Chunking flag
    is_chunked = args.chunks

    if is_chunked:
        original_dataset_path_p = Path(original_dataset_path)
        if not original_dataset_path_p.is_dir():
            print("The joblib file must be a directory when using chunking.")
            return
        joblib_files = [f for f in original_dataset_path_p.iterdir() if f.suffix == ".joblib"]
        if not joblib_files:
            print("No joblib files found in the directory.")
            return
        convert_joblib_to_json_chunked(joblib_files, json_file_path, verbose=True)
        print("Conversion complete.")
        return
        
    # Load the original dataset
    original_dataset = load_dataset(original_dataset_path)

    # Convert the dataset to JSON
    dataset_to_json(original_dataset, json_file_path, verbose=True)

    # Load the dataset back from JSON
    loaded_dataset = load_dataset_from_json(json_file_path)

    # Compare the original and loaded datasets
    if compare_datasets(original_dataset, loaded_dataset):
        print("The conversion worked correctly. The datasets are identical.")
    else:
        print("The conversion did not work correctly. The datasets are not identical.")


if __name__ == "__main__":
    main()
