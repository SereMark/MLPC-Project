
"""
make_segments.py (120 ms → 1.2 s)
Aggregates frame‐level predictions (at 120 ms resolution) into 1.2 s segments.
"""

import argparse
import os
import numpy as np
import pandas as pd
from compute_cost import get_segment_prediction_df, CLASSES

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate 120ms frame-level predictions into 1.2s segments"
    )
    parser.add_argument(
        "--predictions_dir", required=True,
        help="Directory containing .npz files with frame-level arrays for each class"
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Path to write segment-level CSV (filename,onset,CLASSES)"
    )
    args = parser.parse_args()

    # Load all frame-level predictions
    preds = {}
    for fn in os.listdir(args.predictions_dir):
        if not fn.endswith(".npz"):
            continue
        data = np.load(os.path.join(args.predictions_dir, fn))
        # assume each .npz has one array per class name
        preds[fn] = {cls: data[cls] for cls in CLASSES}

    # Aggregate to segments
    df = get_segment_prediction_df(preds, class_names=CLASSES)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote segments to {args.output_csv}")

if __name__ == "__main__":
    main()