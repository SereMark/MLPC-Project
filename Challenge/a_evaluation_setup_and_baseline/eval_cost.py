"""
eval_cost.py (wraps compute_cost.py)
Loads ground truth & prediction CSVs, runs cost evaluation, prints total cost.
"""

import argparse
import pandas as pd
from compute_cost import check_dataframe, total_cost

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CSV predictions against ground truth using custom cost"
    )
    parser.add_argument(
        "--dataset_path", required=True,
        help="Root of MLPC2025_dataset (must contain audio_features/)"
    )
    parser.add_argument(
        "--ground_truth_csv", required=True,
        help="Path to ground_truth.csv"
    )
    parser.add_argument(
        "--predictions_csv", required=True,
        help="Path to predictions.csv"
    )
    args = parser.parse_args()

    gt = pd.read_csv(args.ground_truth_csv)
    pred = pd.read_csv(args.predictions_csv)

    check_dataframe(gt,   args.dataset_path)
    check_dataframe(pred, args.dataset_path)

    total, _ = total_cost(pred, gt)
    print("Total cost:", total)

if __name__ == "__main__":
    main()