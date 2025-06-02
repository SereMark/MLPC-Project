import json
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#                          USER CONFIGURATION
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

SCRIPT_FILE_PATH = Path(__file__).resolve()
SCRIPT_DIRECTORY = SCRIPT_FILE_PATH.parent
PROJECT_ROOT_ASSUMED = SCRIPT_DIRECTORY.parent.parent 

# Paths to dataset, model, and results directory
DATASET_BASE_PATH = PROJECT_ROOT_ASSUMED / "MLPC2025_dataset"
REF_MODEL_PATH = SCRIPT_DIRECTORY / "ref_model.pth"
RESULTS_DIR = SCRIPT_DIRECTORY / "b_core_model_thresholding_results"

# Names for train/validation split files (will be created in DATASET_BASE_PATH)
TRAIN_FILE_LIST_NAME = "train.txt" 
VAL_FILE_LIST_NAME = "val.txt"     

# Data Split Configuration
CREATE_SPLITS_IF_NOT_FOUND = True 
VAL_SPLIT_RATIO = 0.20            
SPLIT_RANDOM_SEED = 42            

# CNN1D Model Configuration (MUST MATCH THE SAVED REF_MODEL_PATH)
MODEL_INPUT_DIM = 942
ORIGINAL_NUM_CLASSES = 58 
ORIGINAL_FULL_CLASS_LIST = [ # list of 58 classes
    "Airplane", "Alarm", "Beep/Bleep", "Bell", "Bicycle", "Bird Chirp", "Bus",
    "Car", "Cat Meow", "Chainsaw", "Clapping", "Cough", "Cow Moo", "Cowbell",
    "Crying", "Dog Bark", "Doorbell", "Drip", "Drums", "Fire", "Footsteps",
    "Guitar", "Hammer", "Helicopter", "Hiccup", "Horn Honk", "Horse Neigh",
    "Insect Buzz", "Jackhammer", "Laughter", "Lawn Mower", "Motorcycle",
    "Piano", "Pig Oink", "Power Drill", "Power Saw", "Rain", "Rooster Crow",
    "Saxophone", "Sewing Machine", "Sheep/Goat Bleat", "Ship/Boat", "Shout",
    "Singing", "Siren", "Sneeze", "Snoring", "Speech", "Stream/River",
    "Thunder", "Train", "Truck", "Trumpet", "Vacuum Cleaner", "Violin",
    "Washing Machine", "Waves", "Wind"
]
CNN1D_MODEL_PARAMS = { # Architecture-specific parameters for CNN1DClassifier
    "cnn_filters_1": 128,
    "cnn_filters_2": 128,
    "kernel_size": 5,
    "dropout_rate": 0.4
}

# Output File Names (within RESULTS_DIR)
OUTPUT_THRESHOLDS_FILENAME = "thresholds.json"
OUTPUT_PREDICTIONS_FILENAME = "ref_predictions_val.csv"

# Optimization Settings
THRESHOLD_OPT_ITERATIONS = 3

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                        END OF USER CONFIGURATION
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# --- Global Constants ---
TARGET_CLASSES = [
    'Speech', 'Shout', 'Chainsaw', 'Jackhammer', 'Lawn Mower',
    'Power Drill', 'Dog Bark', 'Rooster Crow', 'Horn Honk', 'Siren'
]

COST_MATRIX = {
    "Speech":       {"TP": 0, "FP": 1, "TN": 0, "FN": 5},
    "Dog Bark":     {"TP": 0, "FP": 1, "TN": 0, "FN": 5},
    "Rooster Crow": {"TP": 0, "FP": 1, "TN": 0, "FN": 5},
    "Shout":        {"TP": 0, "FP": 2, "TN": 0, "FN": 10},
    "Lawn Mower":   {"TP": 0, "FP": 3, "TN": 0, "FN": 15},
    "Chainsaw":     {"TP": 0, "FP": 3, "TN": 0, "FN": 15},
    "Jackhammer":   {"TP": 0, "FP": 3, "TN": 0, "FN": 15},
    "Power Drill":  {"TP": 0, "FP": 3, "TN": 0, "FN": 15},
    "Horn Honk":    {"TP": 0, "FP": 3, "TN": 0, "FN": 15},
    "Siren":        {"TP": 0, "FP": 3, "TN": 0, "FN": 15},
}

EXPECTED_FEATURE_KEYS = [
    'embeddings', 'melspectrogram', 'mfcc', 'mfcc_delta', 'mfcc_delta2',
    'flatness', 'centroid', 'flux', 'energy', 'power', 'bandwidth',
    'contrast', 'zerocrossingrate'
]
AGGREGATION_FACTOR = 10

class CNN1DClassifier(nn.Module):
    """1D CNN-based classifier for sequence data, adapted from Phase 3."""
    def __init__(self, input_dim: int, num_classes: int, cnn_filters_1: int,
                 cnn_filters_2: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters_1, kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_filters_1), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(cnn_filters_1, cnn_filters_2, kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_filters_2), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.output_conv = nn.Conv1d(cnn_filters_2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.output_conv(x)
        return x.permute(0, 2, 1)

def log_print(message: str, level: str = "INFO"):
    """Prints a timestamped message to the console."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}][{level.upper()}] {message}")

def _generate_and_save_splits(dataset_path: Path, features_subdir_name: str,
                             train_filename: str, val_filename: str,
                             val_ratio: float, seed: int) -> Optional[List[str]]:
    """
    Generates train/validation file stem lists using MultilabelStratifiedShuffleSplit
    based on the TARGET_CLASSES and saves them.
    """
    features_dir = dataset_path / features_subdir_name
    if not features_dir.is_dir():
        log_print(f"Features directory not found for split generation: {features_dir}", "FATAL")
        return None
    
    all_feature_file_stems = sorted([f.stem for f in features_dir.glob("*.npz") if f.is_file()])
    if not all_feature_file_stems:
        log_print(f"No .npz files found in {features_dir} for split generation.", "FATAL")
        return None

    all_audio_filenames_for_gt = [f"{stem}.mp3" for stem in all_feature_file_stems]
    log_print(f"Found {len(all_audio_filenames_for_gt)} files. Generating labels for stratification...")

    # For stratification, we need labels for all files corresponding to TARGET_CLASSES
    gt_all_files_for_strat = get_ground_truth_segments_df(
        all_audio_filenames_for_gt, dataset_path, TARGET_CLASSES, AGGREGATION_FACTOR
    )
    
    if gt_all_files_for_strat.empty:
        log_print("Could not generate ground truth for any files for stratification. Cannot proceed.", "FATAL")
        return None
            
    # Aggregate segment-level labels per file to get a file-level multilabel representation
    file_level_labels_df = gt_all_files_for_strat.groupby('filename')[TARGET_CLASSES].max().reset_index()
    
    filenames_with_labels = file_level_labels_df['filename'].tolist()
    stems_with_labels = [Path(fn).stem for fn in filenames_with_labels]
    
    stem_to_labels_map = {Path(row['filename']).stem: row[TARGET_CLASSES].values 
                              for _, row in file_level_labels_df.iterrows()}
    
    # Prepare X (file stems) and y (multilabel matrix) for splitting from stems_with_labels
    # Ensure only stems that actually have feature files are included.
    X_to_split_stems = [stem for stem in stems_with_labels if stem in all_feature_file_stems] 
    if not X_to_split_stems:
             log_print("No files with both features and processable labels found for stratification. Cannot split.", "FATAL")
             return None
             
    y_to_split_labels = np.array([stem_to_labels_map[stem] for stem in X_to_split_stems])

    if len(X_to_split_stems) < 2 or y_to_split_labels.ndim != 2 or y_to_split_labels.shape[0] != len(X_to_split_stems):
        log_print("Not enough samples or inconsistent label data for stratified split. Cannot split.", "FATAL")
        return None
    
    train_stems: List[str] = []
    val_stems: List[str] = []

    try:
        log_print(f"Performing multilabel stratified split on {len(X_to_split_stems)} files with labels.", "INFO")
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        # We pass indices to split, then use these indices on our X_to_split_stems list
        temp_indices = np.arange(len(X_to_split_stems))
        train_indices, val_indices = next(msss.split(temp_indices, y_to_split_labels))
        
        train_stems = [X_to_split_stems[i] for i in train_indices]
        val_stems = [X_to_split_stems[i] for i in val_indices]
        log_print(f"Successfully performed multilabel stratified split: {len(train_stems)} train, {len(val_stems)} val.", "INFO")
    except Exception as e:
        log_print(f"Error during MultilabelStratifiedShuffleSplit: {e}. This usually means iterative-stratification is not installed or data is unsuitable. Cannot create splits.", "FATAL")
        return None

    val_txt_path = dataset_path / val_filename
    train_txt_path = dataset_path / train_filename

    try:
        with open(val_txt_path, 'w') as f:
            for stem in val_stems: f.write(f"{stem}\n")
        log_print(f"Created {val_txt_path} with {len(val_stems)} entries.")
        with open(train_txt_path, 'w') as f:
            for stem in train_stems: f.write(f"{stem}\n")
        log_print(f"Created {train_txt_path} with {len(train_stems)} entries.")
        return val_stems
    except IOError as e:
        log_print(f"Error writing split files: {e}", "FATAL")
        return None

def load_audio_features_for_file(feature_npz_path: Path, feature_keys: List[str]) -> Optional[np.ndarray]:
    """
    Loads and concatenates specified audio features from an .npz file.
    Aligns features to a common number of time steps by truncating or padding.
    """
    try:
        features_npz = np.load(feature_npz_path)
    except FileNotFoundError:
        log_print(f"Feature file not found: {feature_npz_path}", "WARNING")
        return None
    feature_arrays_list: List[np.ndarray] = []
    available_lengths = []
    valid_keys_for_file = []
    for key in feature_keys:
        if key in features_npz:
            arr = features_npz[key]
            if arr.ndim > 0 and arr.shape[0] > 0:
                available_lengths.append(arr.shape[0])
                valid_keys_for_file.append(key)
    if not available_lengths: return None
    common_time_steps = min(available_lengths)
    if common_time_steps == 0: return None

    for key in valid_keys_for_file:
        arr = features_npz[key]
        arr = arr[:, np.newaxis] if arr.ndim == 1 else arr
        if arr.shape[0] > common_time_steps:
            arr = arr[:common_time_steps, :]
        elif arr.shape[0] < common_time_steps:
            padding_shape = (common_time_steps - arr.shape[0], arr.shape[1])
            padding = np.zeros(padding_shape, dtype=arr.dtype)
            arr = np.vstack((arr, padding))
        feature_arrays_list.append(arr)
        
    if not feature_arrays_list: return None
    try:
        return np.concatenate(feature_arrays_list, axis=-1)
    except ValueError as e:
        log_print(f"Error concatenating features for {feature_npz_path}: {e}", "ERROR")
        return None

def _aggregate_frames_to_segments(arr: np.ndarray, factor: int) -> np.ndarray:
    """Helper to aggregate frame-level data (N, D) to segment-level using max pooling."""
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    N, D = arr.shape
    if N == 0: return np.empty((0, D), dtype=arr.dtype)
    num_segments = math.ceil(N / factor)
    output_arr = np.zeros((num_segments, D), dtype=arr.dtype)
    if np.issubdtype(arr.dtype, np.floating): 
        output_arr = np.full((num_segments, D), -np.inf, dtype=arr.dtype)

    for i in range(num_segments):
        start_frame = i * factor
        end_frame = min((i + 1) * factor, N)
        if start_frame < end_frame:
             output_arr[i] = np.max(arr[start_frame:end_frame, :], axis=0)
    return output_arr

def get_ground_truth_segments_df(audio_filenames_with_ext: Iterable[str], dataset_base_path: Path,
                                target_classes_list: List[str], aggregation_frame_factor: int) -> pd.DataFrame:
    """
    Loads ground truth labels, aggregates frame-level scores to segment-level (max-pooling),
    and then binarizes segment scores (>0 indicates presence).
    """
    output_rows = []
    for audio_fname in audio_filenames_with_ext:
        audio_file_stem = Path(audio_fname).stem
        label_file_path = dataset_base_path / 'labels' / f"{audio_file_stem}_labels.npz"
        if not label_file_path.exists():
            log_print(f"Label file not found: {label_file_path}", "WARNING")
            continue
        try:
            label_data_npz = np.load(label_file_path)
            frame_level_scores_per_class = []
            min_frames_this_file = -1

            for cls_name in target_classes_list:
                if cls_name in label_data_npz:
                    class_array_from_npz = label_data_npz[cls_name]
                    class_scores_1d = None
                    if class_array_from_npz.ndim == 2:
                        class_scores_1d = class_array_from_npz.mean(axis=-1)
                    elif class_array_from_npz.ndim == 1:
                        class_scores_1d = class_array_from_npz
                    
                    if class_scores_1d is not None and class_scores_1d.shape[0] > 0 :
                        if min_frames_this_file == -1 or len(class_scores_1d) < min_frames_this_file:
                            min_frames_this_file = len(class_scores_1d)
                    elif class_scores_1d is not None and class_scores_1d.shape[0] == 0:
                        class_scores_1d = None 
                    frame_level_scores_per_class.append(class_scores_1d)
                else:
                    frame_level_scores_per_class.append(None) 
            
            if min_frames_this_file <= 0 :
                continue

            aligned_frame_scores_list = []
            for scores_array in frame_level_scores_per_class:
                if scores_array is not None:
                    aligned_frame_scores_list.append(scores_array[:min_frames_this_file])
                else: 
                    aligned_frame_scores_list.append(np.zeros(min_frames_this_file))
            
            frame_scores_matrix = np.stack(aligned_frame_scores_list, axis=1)
            segment_aggregated_scores = _aggregate_frames_to_segments(frame_scores_matrix, factor=aggregation_frame_factor)
            segment_binary_labels = (segment_aggregated_scores > 0).astype(int)
            
            for seg_idx, binary_labels_row in enumerate(segment_binary_labels):
                onset_time_sec = round(seg_idx * 1.2, 1)
                output_rows.append([audio_fname, onset_time_sec] + binary_labels_row.tolist())
        except Exception as e:
            log_print(f"Error processing GT for {audio_fname}: {e}", "ERROR")
            
    return pd.DataFrame(output_rows, columns=["filename", "onset"] + target_classes_list)

def get_segment_probabilities_df(frame_level_file_probs: Dict[str, Dict[str, np.ndarray]],
                                target_classes_list: List[str], aggregation_frame_factor: int) -> pd.DataFrame:
    """Aggregates per-file, per-class frame-level probabilities to segment-level (max-pooling)."""
    output_rows = []
    for audio_fname, class_frame_data_dict in frame_level_file_probs.items():
        prob_arrays_for_stacking = []
        min_frames_this_file = -1
        for cls_name in target_classes_list:
            class_prob_array = class_frame_data_dict.get(cls_name)
            if class_prob_array is not None and class_prob_array.ndim == 1 and class_prob_array.shape[0] > 0:
                if min_frames_this_file == -1 or len(class_prob_array) < min_frames_this_file:
                    min_frames_this_file = len(class_prob_array)
                prob_arrays_for_stacking.append(class_prob_array)
            else: 
                prob_arrays_for_stacking.append(None) 
        
        if min_frames_this_file <= 0: 
            continue

        aligned_prob_arrays_list = []
        for prob_array in prob_arrays_for_stacking:
            if prob_array is not None:
                aligned_prob_arrays_list.append(prob_array[:min_frames_this_file])
            else: 
                aligned_prob_arrays_list.append(np.zeros(min_frames_this_file))
                
        frame_probs_matrix = np.stack(aligned_prob_arrays_list, axis=1)
        segment_aggregated_max_probs = _aggregate_frames_to_segments(frame_probs_matrix, factor=aggregation_frame_factor)
        
        for seg_idx, max_probs_row in enumerate(segment_aggregated_max_probs):
            onset_time_sec = round(seg_idx * 1.2, 1)
            output_rows.append([audio_fname, onset_time_sec] + max_probs_row.tolist())
            
    return pd.DataFrame(output_rows, columns=["filename", "onset"] + target_classes_list)

def calculate_total_cost(predictions_binary_df: pd.DataFrame, ground_truth_binary_df: pd.DataFrame,
                         target_classes_list: List[str], class_cost_matrix: Dict) -> Tuple[float, Dict]:
    """
    Computes total cost based on predictions and ground truth, using the provided cost matrix.
    Metrics are calculated on data merged by filename and onset for robustness.
    """
    preds_sorted = predictions_binary_df.sort_values(by=['filename', 'onset']).reset_index(drop=True)
    gt_sorted = ground_truth_binary_df.sort_values(by=['filename', 'onset']).reset_index(drop=True)
    
    merged_eval_df = pd.merge(
        preds_sorted, gt_sorted, on=["filename", "onset"], suffixes=("_pred", "_true"), how="inner"
    )
    
    if merged_eval_df.empty:
        log_print("Merged DataFrame for cost calculation is empty. Cost is 0.", "WARNING")
        return 0.0, {cls: {"cost": 0, "TP_norm": 0, "FP_norm": 0, "FN_norm": 0, "TN_norm": 0} for cls in target_classes_list}

    class_metrics_summary = {}
    overall_total_cost_sum = 0.0
    total_segments_for_norm = len(merged_eval_df)

    for cls_name in target_classes_list:
        pred_col_name = cls_name + "_pred"
        true_col_name = cls_name + "_true"

        if pred_col_name not in merged_eval_df.columns or true_col_name not in merged_eval_df.columns:
            log_print(f"Columns for class '{cls_name}' missing in merged data. Skipping cost for this class.", "WARNING")
            class_metrics_summary[cls_name] = {"cost": 0, "TP_norm": 0, "FP_norm": 0, "FN_norm": 0, "TN_norm": 0}
            continue

        y_pred_series = merged_eval_df[pred_col_name].astype(int)
        y_true_series = merged_eval_df[true_col_name].astype(int)
        
        tp_count = ((y_pred_series == 1) & (y_true_series == 1)).sum()
        fp_count = ((y_pred_series == 1) & (y_true_series == 0)).sum()
        fn_count = ((y_pred_series == 0) & (y_true_series == 1)).sum()
        
        tp_normalized = (tp_count / total_segments_for_norm) * 50 if total_segments_for_norm > 0 else 0
        fp_normalized = (fp_count / total_segments_for_norm) * 50 if total_segments_for_norm > 0 else 0
        tn_normalized = 0 
        fn_normalized = (fn_count / total_segments_for_norm) * 50 if total_segments_for_norm > 0 else 0
        
        costs_for_this_class = class_cost_matrix[cls_name]
        current_class_total_cost = (
            costs_for_this_class["TP"] * tp_normalized +
            costs_for_this_class["FP"] * fp_normalized +
            costs_for_this_class["TN"] * tn_normalized +
            costs_for_this_class["FN"] * fn_normalized
        )
        class_metrics_summary[cls_name] = {
            "cost": current_class_total_cost,
            "TP_norm": tp_normalized, "FP_norm": fp_normalized,
            "TN_norm": tn_normalized, "FN_norm": fn_normalized
        }
        overall_total_cost_sum += current_class_total_cost
        
    return overall_total_cost_sum, class_metrics_summary

def _apply_thresholds_to_df(df_segment_probs: pd.DataFrame, thresholds_dict: Dict[str, float],
                           class_list: List[str]) -> pd.DataFrame:
    """Applies per-class thresholds to a DataFrame of segment probabilities to get binary predictions."""
    df_binary_preds = df_segment_probs[['filename', 'onset']].copy()
    for cls_name in class_list:
        if cls_name in df_segment_probs.columns:
            probs_for_class = df_segment_probs[cls_name]
            threshold_for_class = thresholds_dict.get(cls_name, 0.5) 
            df_binary_preds[cls_name] = (probs_for_class >= threshold_for_class).astype(int)
        else:
            log_print(f"Class '{cls_name}' not found in probability DataFrame. Predicting all zeros.", "WARNING")
            df_binary_preds[cls_name] = 0 
    return df_binary_preds

def run_core_sed_pipeline():
    """Main orchestrator."""
    script_start_time = time.time()
    log_print(f"Starting: Core Model & Thresholds")
    log_print(f"Timestamp: {time.ctime(script_start_time)}\n")

    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {current_device}\n")

    # --- 1. Ensure Train/Validation Splits Exist ---
    actual_val_file_list_path = DATASET_BASE_PATH / VAL_FILE_LIST_NAME
    if CREATE_SPLITS_IF_NOT_FOUND and not actual_val_file_list_path.exists():
        log_print(f"{actual_val_file_list_path} not found. Generating train/val splits from {DATASET_BASE_PATH}.", "INFO")
        val_stems_generated = _generate_and_save_splits(
            DATASET_BASE_PATH, "audio_features", TRAIN_FILE_LIST_NAME, VAL_FILE_LIST_NAME,
            VAL_SPLIT_RATIO, SPLIT_RANDOM_SEED
        )
        if not val_stems_generated:
            log_print(f"Failed to generate split files. Exiting.", "FATAL")
            return
    elif not actual_val_file_list_path.exists():
        log_print(f"{actual_val_file_list_path} not found and CREATE_SPLITS_IF_NOT_FOUND is False. Exiting.", "FATAL")
        return
    
    # --- 2. Verify Model Parameters and Load Model ---
    log_print(f"Using hardcoded model parameters for CNN1D.")
    if MODEL_INPUT_DIM != 942:
         log_print(f"Configured MODEL_INPUT_DIM ({MODEL_INPUT_DIM}) differs from expected Phase 3 log value (942).", "WARNING")
    if ORIGINAL_NUM_CLASSES != 58:
         log_print(f"Configured ORIGINAL_NUM_CLASSES ({ORIGINAL_NUM_CLASSES}) differs from expected Phase 3 log value (58).", "WARNING")
    if len(ORIGINAL_FULL_CLASS_LIST) != ORIGINAL_NUM_CLASSES:
        log_print(f"Length of ORIGINAL_FULL_CLASS_LIST ({len(ORIGINAL_FULL_CLASS_LIST)}) does not match ORIGINAL_NUM_CLASSES ({ORIGINAL_NUM_CLASSES}). Exiting.", "FATAL")
        return

    log_print(f"Model originally trained for {ORIGINAL_NUM_CLASSES} classes.")
    log_print(f"Input dimension expected by model: {MODEL_INPUT_DIM}.")

    target_class_model_indices = []
    for target_cls in TARGET_CLASSES:
        try:
            idx = ORIGINAL_FULL_CLASS_LIST.index(target_cls)
            target_class_model_indices.append(idx)
        except ValueError:
            log_print(f"Target class '{target_cls}' NOT FOUND in ORIGINAL_FULL_CLASS_LIST. Will output zeros for it.", "ERROR")
            target_class_model_indices.append(-1) # Mark as missing

    log_print(f"\nLoading pre-trained 'CNN1D' model from: {REF_MODEL_PATH}")
    model_init_actual_params = {
        "input_dim": MODEL_INPUT_DIM,
        "num_classes": ORIGINAL_NUM_CLASSES, # Model expects original number of output classes
        **CNN1D_MODEL_PARAMS
    }

    try:
        sed_model = CNN1DClassifier(**model_init_actual_params).to(current_device)
        sed_model.load_state_dict(torch.load(REF_MODEL_PATH, map_location=current_device))
        sed_model.eval()
        log_print("Model loaded successfully.")
    except FileNotFoundError:
        log_print(f"Model checkpoint file not found at '{REF_MODEL_PATH}'. Exiting.", "FATAL")
        return
    except Exception as e:
        log_print(f"Error loading model state_dict: {e}. Check model params and checkpoint path. Exiting.", "FATAL")
        return

    # --- 3. Process Validation Files: Get Frame-Level Probabilities ---
    log_print(f"\nProcessing validation files listed in: {actual_val_file_list_path}")
    val_file_stems_list = []
    try:
        with open(actual_val_file_list_path, 'r') as f:
            for line in f:
                stem = line.strip()
                if stem: val_file_stems_list.append(Path(stem).stem)
    except FileNotFoundError:
        log_print(f"Validation file list '{actual_val_file_list_path}' not found. Exiting.", "FATAL")
        return
    if not val_file_stems_list:
        log_print(f"Validation file list '{actual_val_file_list_path}' is empty. Exiting.", "FATAL")
        return

    validation_audio_filenames_with_ext = [f"{stem}.mp3" for stem in val_file_stems_list] # Assuming .mp3 audio files
    log_print(f"Found {len(validation_audio_filenames_with_ext)} validation file entries to process.")

    all_audio_frame_probabilities: Dict[str, Dict[str, np.ndarray]] = {}
    processed_files_count = 0
    for audio_file_stem in val_file_stems_list:
        audio_filename = f"{audio_file_stem}.mp3"
        feature_file_path = DATASET_BASE_PATH / "audio_features" / f"{audio_file_stem}.npz"
        
        concatenated_features = load_audio_features_for_file(feature_file_path, EXPECTED_FEATURE_KEYS)
        if concatenated_features is None or concatenated_features.shape[0] == 0:
            all_audio_frame_probabilities[audio_filename] = {cls: np.array([]) for cls in TARGET_CLASSES}
            continue # Skip if features can't be loaded or are empty
            
        features_tensor = torch.from_numpy(concatenated_features).float().unsqueeze(0).to(current_device)
        
        with torch.no_grad():
            model_output_logits = sed_model(features_tensor) # Shape: (1, Time, ORIGINAL_NUM_CLASSES)
        
        current_file_class_probs: Dict[str, np.ndarray] = {}
        for target_idx, model_output_idx in enumerate(target_class_model_indices):
            class_name = TARGET_CLASSES[target_idx]
            if model_output_idx != -1: # If this target class is mapped to a model output
                class_probs = torch.sigmoid(model_output_logits[0, :, model_output_idx]).cpu().numpy()
                current_file_class_probs[class_name] = class_probs
            else: # Target class not in model's original output, assign zero probabilities
                current_file_class_probs[class_name] = np.zeros(concatenated_features.shape[0])
        
        all_audio_frame_probabilities[audio_filename] = current_file_class_probs
        processed_files_count +=1
        
    log_print(f"Processed features and obtained model probabilities for {processed_files_count} / {len(val_file_stems_list)} validation files.")
    if processed_files_count == 0:
        log_print("No validation files could be processed for features. Cannot continue.", "FATAL")
        return

    # --- 4. Aggregate Probabilities to Segment Level ---
    log_print("\nAggregating frame probabilities to 1.2s segment probabilities (max-pooling)...")
    df_segment_max_probs_val = get_segment_probabilities_df(
        all_audio_frame_probabilities, TARGET_CLASSES, AGGREGATION_FACTOR
    )
    if df_segment_max_probs_val.empty:
        log_print("Segment probability DataFrame is empty after aggregation. Cannot continue.", "FATAL")
        return
    df_segment_max_probs_val = df_segment_max_probs_val.sort_values(by=['filename', 'onset']).reset_index(drop=True)

    # --- 5. Generate Ground Truth for Validation Set ---
    log_print("\nGenerating 1.2s segment-level ground truth for validation set...")
    df_ground_truth_val = get_ground_truth_segments_df(
        validation_audio_filenames_with_ext, DATASET_BASE_PATH, TARGET_CLASSES, AGGREGATION_FACTOR
    )
    if df_ground_truth_val.empty:
        log_print("Ground truth DataFrame is empty. Check label files and val.txt. Cannot continue.", "FATAL")
        return
    df_ground_truth_val = df_ground_truth_val.sort_values(by=['filename', 'onset']).reset_index(drop=True)

    # Persist ground-truth CSV for the eval script
    gt_csv_path = DATASET_BASE_PATH / "ground_truth_val.csv"
    df_ground_truth_val.to_csv(gt_csv_path, index=False)
    log_print(f"Saved ground truth for validation set to: {gt_csv_path}")

    # --- 6. Optimize Per-Class Thresholds ---
    log_print("\nOptimizing per-class thresholds using the cost matrix...")
    current_optimized_thresholds: Dict[str, float] = {cls: 0.5 for cls in TARGET_CLASSES}
    df_initial_binary_preds = _apply_thresholds_to_df(df_segment_max_probs_val, current_optimized_thresholds, TARGET_CLASSES)
    min_overall_cost, _ = calculate_total_cost(df_initial_binary_preds, df_ground_truth_val, TARGET_CLASSES, COST_MATRIX)
    log_print(f"  Initial cost (all thresholds at 0.5): {min_overall_cost:.4f}")

    threshold_sweep_values = np.arange(0.05, 1.0, 0.05) # Candidate thresholds from 0.05 to 0.95

    for iteration_num in range(THRESHOLD_OPT_ITERATIONS):
        log_print(f"  Threshold Optimization Iteration {iteration_num + 1}/{THRESHOLD_OPT_ITERATIONS}")
        cost_improved_this_iteration = False
        cost_at_iteration_start = min_overall_cost

        for class_to_opt in TARGET_CLASSES:
            best_threshold_for_this_class_in_sweep = current_optimized_thresholds[class_to_opt]
            cost_before_sweeping_this_class = min_overall_cost 

            for candidate_thresh in threshold_sweep_values:
                temp_thresholds_for_eval = current_optimized_thresholds.copy()
                temp_thresholds_for_eval[class_to_opt] = candidate_thresh
                df_temp_binary_preds = _apply_thresholds_to_df(df_segment_max_probs_val, temp_thresholds_for_eval, TARGET_CLASSES)
                cost_with_candidate_thresh, _ = calculate_total_cost(df_temp_binary_preds, df_ground_truth_val, TARGET_CLASSES, COST_MATRIX)

                if cost_with_candidate_thresh < cost_before_sweeping_this_class:
                    cost_before_sweeping_this_class = cost_with_candidate_thresh
                    best_threshold_for_this_class_in_sweep = candidate_thresh
            
            if abs(current_optimized_thresholds[class_to_opt] - best_threshold_for_this_class_in_sweep) > 1e-6 : # Check for actual change
                current_optimized_thresholds[class_to_opt] = best_threshold_for_this_class_in_sweep
                min_overall_cost = cost_before_sweeping_this_class 
        
        if min_overall_cost < cost_at_iteration_start:
            cost_improved_this_iteration = True
        log_print(f"  Cost after iteration {iteration_num + 1}: {min_overall_cost:.4f}")
        if not cost_improved_this_iteration and iteration_num > 0: # Stop if an iteration yields no change (after the first pass)
            log_print("  No improvement in overall cost this iteration. Optimization likely converged.")
            break
            
    log_print(f"\nFinal Optimized Thresholds: {current_optimized_thresholds}")
    thresholds_output_path = results_path / OUTPUT_THRESHOLDS_FILENAME
    with open(thresholds_output_path, "w") as f:
        json.dump({k: float(v) for k, v in current_optimized_thresholds.items()}, f, indent=4)
    log_print(f"Saved optimized thresholds to: {thresholds_output_path}")

    # --- 7. Generate Final Validation Predictions & Report Cost ---
    log_print("\nGenerating final validation predictions using optimized thresholds...")
    df_final_val_predictions_binary = _apply_thresholds_to_df(
        df_segment_max_probs_val, current_optimized_thresholds, TARGET_CLASSES
    )
    final_predictions_output_path = results_path / OUTPUT_PREDICTIONS_FILENAME
    df_final_val_predictions_binary.to_csv(final_predictions_output_path, index=False)
    log_print(f"Saved final binary predictions for validation set to: {final_predictions_output_path}")

    # --- 8. Run Inference on the Hidden Test Set ---
    TEST_DATASET_PATH = PROJECT_ROOT_ASSUMED / "MLPC2025_test"
    test_feat_dir     = TEST_DATASET_PATH / "audio_features"
    test_stems        = sorted(p.stem for p in test_feat_dir.glob("*.npz"))
    frame_probs_test: Dict[str, Dict[str, np.ndarray]] = {}
    
    for stem in test_stems:
        feats = load_audio_features_for_file(test_feat_dir / f"{stem}.npz",
                                             EXPECTED_FEATURE_KEYS)
        if feats is None or feats.shape[0] == 0:
            frame_probs_test[f"{stem}.mp3"] = {c: np.array([]) for c in TARGET_CLASSES}
            continue
        with torch.no_grad():
            logits = sed_model(torch.from_numpy(feats).float()
                                             .unsqueeze(0).to(current_device))
        per_file = {}
        for tgt_i, mdl_i in enumerate(target_class_model_indices):
            cls = TARGET_CLASSES[tgt_i]
            per_file[cls] = (torch.sigmoid(logits[0, :, mdl_i]).cpu().numpy()
                             if mdl_i != -1 else
                             np.zeros(feats.shape[0]))
        frame_probs_test[f"{stem}.mp3"] = per_file
    
    # aggregate → threshold → CSV
    df_seg_probs_test = get_segment_probabilities_df(frame_probs_test,
                                                     TARGET_CLASSES,
                                                     AGGREGATION_FACTOR)
    df_test_bin = _apply_thresholds_to_df(df_seg_probs_test,
                                          current_optimized_thresholds,
                                          TARGET_CLASSES)
    test_csv_path = results_path / "predictions_test.csv"
    df_test_bin.to_csv(test_csv_path, index=False)
    log_print(f"Saved TEST predictions to: {test_csv_path}")

    final_reported_cost, final_metrics_breakdown = calculate_total_cost(
        df_final_val_predictions_binary, df_ground_truth_val, TARGET_CLASSES, COST_MATRIX
    )
    log_print(f"\n--- FINAL VALIDATION COST ---", "RESULT")
    log_print(f"===> Total Cost on Validation Set: {final_reported_cost:.4f} <===", "RESULT")
    log_print("\nCost breakdown per class (normalized per minute equivalent):", "RESULT")
    for cls_name in TARGET_CLASSES:
        m = final_metrics_breakdown[cls_name]
        log_print(f"  {cls_name:<13} Cost={m['cost']:.2f} "
                f"(FPn={m['FP_norm']:.2f}, FNn={m['FN_norm']:.2f})", "RESULT")

    total_script_time = time.time() - script_start_time
    log_print(f"\n--- Processing completed in {total_script_time:.2f} seconds ---", "RESULT")
    log_print(f"All outputs are available in: {results_path.resolve()}", "RESULT")

if __name__ == "__main__":
    run_core_sed_pipeline()