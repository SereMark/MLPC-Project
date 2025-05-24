import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Iterator
import random
import time
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import UndefinedMetricWarning 
import warnings

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "MLPC2025_dataset" 

FEATURE_DIR = BASE_DIR / "audio_features"
LABEL_DIR = BASE_DIR / "labels"

OUTPUT_DIR = SCRIPT_DIR / "experiments_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

LOG_FILE_PATH = OUTPUT_DIR / "experiment_log.txt"

# Training Hyperparameters
NUM_EPOCHS_HYPERPARAM_SEARCH = 15 
NUM_EPOCHS_FINAL_TRAIN = 30     
BATCH_SIZE = 32 
RANDOM_STATE = 42

# --- Global Variables (to be initialized by function) ---
CLASS_NAMES: List[str] = []
FULL_FILE_LIST: List[str] = []
EXPECTED_FEATURE_KEYS = [ 
    'embeddings', 'melspectrogram', 'mfcc', 'mfcc_delta', 'mfcc_delta2',
    'flatness', 'centroid', 'flux', 'energy', 'power', 'bandwidth',
    'contrast', 'zerocrossingrate'
]
INPUT_DIM: int = 0 
NUM_CLASSES: int = 0

# --- Seed Management & Warnings ---
def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, module="sklearn.metrics._classification")

# --- Logging ---
def log_message(message: str):
    """Log a message with timestamp to both console and log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")
    except Exception as e:
        print(f"[{timestamp}] Error writing to log file ({LOG_FILE_PATH}): {e}")

# --- Dataset Initialization and Verification ---
def initialize_global_dataset_vars() -> bool:
    """Initialize global dataset variables and verify data integrity."""
    global CLASS_NAMES, FULL_FILE_LIST, NUM_CLASSES, INPUT_DIM
    
    log_message("Initializing global dataset variables...")
    
    # Check critical paths
    critical_paths = {
        "Base Dataset Dir": BASE_DIR, 
        "Feature Dir": FEATURE_DIR, 
        "Label Dir": LABEL_DIR
    }
    
    for name, path_to_check in critical_paths.items():
        if not path_to_check.exists():
            log_message(f"CRITICAL ERROR: {name} '{path_to_check}' not found. Please check dataset location.")
            return False
    
    # Load class names
    all_label_files = sorted(list(LABEL_DIR.glob("*_labels.npz")))
    if not all_label_files:
        log_message(f"CRITICAL ERROR: No label files in {LABEL_DIR}.")
        return False
    
    try:
        with np.load(all_label_files[0]) as sf:
            CLASS_NAMES = sorted(list(sf.keys()))
        
        NUM_CLASSES = len(CLASS_NAMES)
        if NUM_CLASSES == 0:
            log_message(f"CRITICAL: No classes in {all_label_files[0]}.")
            return False
        
        log_message(f"Loaded {NUM_CLASSES} classes (e.g., {CLASS_NAMES[:min(3, NUM_CLASSES)]}...).")
    except Exception as e:
        log_message(f"CRITICAL: Load classes fail {all_label_files[0]}: {e}")
        return False
    
    # Get feature files list
    FULL_FILE_LIST = sorted([f.stem for f in FEATURE_DIR.glob("*.npz") if f.is_file()])
    if not FULL_FILE_LIST:
        log_message(f"CRITICAL: No feature files in {FEATURE_DIR}.")
        return False
    
    log_message(f"Found {len(FULL_FILE_LIST)} feature files.")
    
    # Calculate input dimension
    try:
        with np.load(FEATURE_DIR / f"{FULL_FILE_LIST[0]}.npz") as f_npz:
            calc_in_dim = sum(
                f_npz[k].shape[1] if f_npz[k].ndim > 1 else 1 
                for k in EXPECTED_FEATURE_KEYS if k in f_npz
            )
            
            if not all(k in f_npz for k in EXPECTED_FEATURE_KEYS):
                log_message(f"CRITICAL: Key missing in {FULL_FILE_LIST[0]}.npz.")
                return False
            
            INPUT_DIM = calc_in_dim
            
        log_message(f"Calculated INPUT_DIM: {INPUT_DIM} from {len(EXPECTED_FEATURE_KEYS)} features.")
        
        if INPUT_DIM == 0:
            log_message(f"CRITICAL: INPUT_DIM is 0.")
            return False
            
    except Exception as e:
        log_message(f"CRITICAL: Calc INPUT_DIM fail {FULL_FILE_LIST[0]}.npz: {e}")
        return False
        
    return True

# --- PyTorch Dataset & Dataloader Components ---
class AudioFrameDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for audio frame-level features and labels."""
    
    def __init__(self, feature_dir: Path, label_dir: Path, file_ids: List[str], 
                 class_names_arg: List[str], feature_keys_arg: List[str]):
        self.feature_dir: Path = feature_dir
        self.label_dir: Path = label_dir
        self.class_names: List[str] = class_names_arg 
        self.feature_keys_to_concat: List[str] = feature_keys_arg
        self.processed_file_infos: List[Dict[str, Any]] = []
        
        for file_id in file_ids:
            feature_path = self.feature_dir / f"{file_id}.npz"
            label_path = self.label_dir / f"{file_id}_labels.npz"
            
            if not (feature_path.exists() and label_path.exists()):
                continue
                
            try:
                with np.load(feature_path) as features_npz:
                    ref_key = next((fk for fk in self.feature_keys_to_concat if fk in features_npz), None)
                    if ref_key is None or features_npz[ref_key].shape[0] == 0:
                        continue
                        
                    self.processed_file_infos.append({
                        "file_id": file_id,
                        "length": features_npz[ref_key].shape[0]
                    })
            except Exception:
                continue
                
        if not self.processed_file_infos:
            log_message("AudioFrameDataset: WARNING - No valid files loaded.")

    def __len__(self) -> int:
        return len(self.processed_file_infos)
        
    def get_length(self, idx: int) -> int:
        return self.processed_file_infos[idx]["length"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_info = self.processed_file_infos[idx]
        file_id = file_info["file_id"]
        
        try: 
            features_npz = np.load(self.feature_dir / f"{file_id}.npz")
            label_npz = np.load(self.label_dir / f"{file_id}_labels.npz")
        except Exception as e:
            raise IOError(f"Error loading {file_id}: {e}")
        
        # Find common time steps across features
        available_lengths = [
            features_npz[key].shape[0] 
            for key in self.feature_keys_to_concat 
            if key in features_npz and hasattr(features_npz[key], 'shape')
        ]
        
        if not available_lengths:
            raise ValueError(f"No valid features in {file_id}.npz.")
            
        common_time_steps = min(available_lengths)
        if common_time_steps == 0:
            raise ValueError(f"Sequence length 0 for {file_id}.npz.")

        # Concatenate feature arrays
        feature_arrays_list: List[np.ndarray] = []
        for key in self.feature_keys_to_concat:
            if key not in features_npz:
                raise KeyError(f"Key '{key}' missing in {file_id}.npz.")
                
            arr = features_npz[key]
            arr = arr[:, np.newaxis] if arr.ndim == 1 else arr
            
            if arr.shape[0] != common_time_steps:
                arr_aligned = np.zeros((common_time_steps, arr.shape[1]), dtype=arr.dtype)
                len_to_copy = min(arr.shape[0], common_time_steps)
                arr_aligned[:len_to_copy] = arr[:len_to_copy]
                arr = arr_aligned
                
            feature_arrays_list.append(arr)
        
        # Prepare feature and label tensors
        feature_data_np = np.concatenate(feature_arrays_list, axis=-1)
        label_data_np = np.zeros((common_time_steps, len(self.class_names)), dtype=np.float32)
        
        for i, class_name_iter in enumerate(self.class_names):
            if class_name_iter in label_npz:
                cl_all = label_npz[class_name_iter]
                lbl_al = np.zeros((common_time_steps, cl_all.shape[1]), dtype=cl_all.dtype)
                cp_len = min(cl_all.shape[0], common_time_steps)
                lbl_al[:cp_len] = cl_all[:cp_len]
                label_data_np[:, i] = (lbl_al.mean(axis=-1) > 0.5).astype(np.float32)
                
        return torch.from_numpy(feature_data_np).float(), torch.from_numpy(label_data_np).float()

class BucketBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups samples of similar lengths together."""
    
    def __init__(self, dataset: AudioFrameDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_batches = shuffle
        
        # Filter out empty sequences and sort by length
        indices = [i for i in range(len(dataset)) if dataset.get_length(i) > 0] 
        self.sorted_indices = sorted(indices, key=lambda i: dataset.get_length(i))
        
        # Create batches
        self.batches: List[List[int]] = [
            self.sorted_indices[i:i+batch_size] 
            for i in range(0, len(self.sorted_indices), batch_size)
        ]
        
        if self.shuffle_batches:
            random.shuffle(self.batches)
            
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle_batches:
            random.shuffle(self.batches)
        for batch_indices in self.batches:
            yield batch_indices
            
    def __len__(self) -> int:
        return len(self.batches)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function that handles variable-length sequences."""
    if not batch:
        return torch.tensor([]), torch.tensor([])
        
    feat_list, lbl_list = zip(*batch)
    valid_feats = [f for f in feat_list if f.numel() > 0]
    
    if not valid_feats:
        return torch.empty(len(feat_list), 0, 0), torch.empty(len(lbl_list), 0, 0)
        
    max_ts = max(f.shape[0] for f in valid_feats)
    nf = valid_feats[0].shape[1]
    nlc = lbl_list[0].shape[1] if lbl_list and lbl_list[0].numel() > 0 else NUM_CLASSES
    
    pad_f = torch.zeros(len(feat_list), max_ts, nf, dtype=torch.float32)
    pad_l = torch.zeros(len(lbl_list), max_ts, nlc, dtype=torch.float32)
    
    for i, ft in enumerate(feat_list): 
        if ft.numel() > 0:
            pad_f[i, :ft.shape[0], :] = ft
            
    for i, lt in enumerate(lbl_list): 
        if lt.numel() > 0:
            pad_l[i, :lt.shape[0], :] = lt
            
    return pad_f, pad_l

def create_data_splits(
    file_ids: List[str],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split data into train, validation, and test sets."""
    if not(0 < test_ratio < 1 and 0 < val_ratio < 1 and (test_ratio + val_ratio) < 1):
        raise ValueError("Ratios invalid.")
        
    tv_ids, tst_ids = train_test_split(file_ids, test_size=test_ratio, random_state=random_state)
    rel_val_r = val_ratio / (1.0 - test_ratio)
    trn_ids, val_ids = train_test_split(tv_ids, test_size=rel_val_r, random_state=random_state)
    
    return trn_ids, val_ids, tst_ids

def calculate_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_fn_bce: nn.Module
) -> Tuple[torch.Tensor, float]:
    """Calculate loss and balanced accuracy for predictions."""
    flat_logits = logits.reshape(-1, NUM_CLASSES)
    flat_labels = labels.reshape(-1, NUM_CLASSES)
    
    loss = loss_fn_bce(flat_logits, flat_labels).mean()
    
    pred_b = (torch.sigmoid(flat_logits) > 0.5).cpu().numpy()
    true_b = flat_labels.cpu().numpy()
    
    cb_accs: List[float] = []
    for i in range(NUM_CLASSES):
        t_c_l = true_b[:, i]
        p_c_l = pred_b[:, i]
        uniq_t_l = np.unique(t_c_l)
        
        if len(uniq_t_l) == 1:
            bacc = 1.0 if np.all(t_c_l == p_c_l) else 0.0
        else:
            bacc = balanced_accuracy_score(t_c_l, p_c_l) 
            
        cb_accs.append(bacc)
        
    return loss, float(np.mean(cb_accs) if cb_accs else 0.0)

# --- Model Architectures ---
class SimpleTimeStepClassifier(nn.Module):
    """Simple MLP classifier for timestep data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        x_f = x.reshape(b * t, -1)
        x_p = self.relu(self.fc1(x_f))
        o_f = self.fc2(x_p)
        return o_f.reshape(b, t, -1)

class LSTMClassifier(nn.Module):
    """LSTM-based classifier for sequence data."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                num_classes: int, dropout_rate: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(x)
        o_d = self.dropout(o)
        b, t, h_d = o_d.shape
        o_f = o_d.reshape(b * t, h_d)
        l_f = self.fc(o_f)
        return l_f.reshape(b, t, -1)

class CNN1DClassifier(nn.Module):
    """1D CNN-based classifier for sequence data."""
    
    def __init__(self, input_dim: int, num_classes: int, cnn_filters_1: int,
                cnn_filters_2: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters_1, kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_filters_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(cnn_filters_1, cnn_filters_2, kernel_size, padding='same'),
            nn.BatchNorm1d(cnn_filters_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_conv = nn.Conv1d(cnn_filters_2, num_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.output_conv(x)
        return x.permute(0, 2, 1)

# --- Training & Evaluation Routines ---
def train_epoch_fn(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer,
    loss_fn: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    epoch_total_loss = 0.0
    epoch_total_accuracy = 0.0
    num_batches = len(dataloader)

    if num_batches == 0: 
        log_message("Warning: Training dataloader is empty.")
        return 0.0, 0.0
        
    for features_batch, labels_batch in dataloader:
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        optimizer.zero_grad()
        logits_batch = model(features_batch)
        
        current_batch_loss, current_batch_accuracy = calculate_metrics(
            logits_batch, labels_batch, loss_fn
        )
        
        if torch.isnan(current_batch_loss) or torch.isinf(current_batch_loss):
            log_message("Warning: NaN/Inf loss in training batch. Skipping update for this batch.")
            continue 
        
        current_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_total_loss += current_batch_loss.item()
        epoch_total_accuracy += current_batch_accuracy 
        
    avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
    avg_epoch_accuracy = epoch_total_accuracy / num_batches if num_batches > 0 else 0.0
    
    return avg_epoch_loss, avg_epoch_accuracy

def evaluate_model_fn(
    model: nn.Module, 
    dataloader: DataLoader,
    loss_fn: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on dataloader."""
    model.eval()
    eval_total_loss = 0.0
    eval_total_accuracy = 0.0
    
    valid_batches_for_loss_avg = 0
    valid_batches_for_acc_avg = 0
    
    num_batches = len(dataloader)
    if num_batches == 0: 
        log_message("Warning: Evaluation dataloader is empty.")
        return 0.0, 0.0
        
    with torch.no_grad():
        for features_batch, labels_batch in dataloader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            logits_batch = model(features_batch)
            current_batch_loss, current_batch_accuracy = calculate_metrics(
                logits_batch, labels_batch, loss_fn
            )
            
            if not (torch.isnan(current_batch_loss) or torch.isinf(current_batch_loss)):
                eval_total_loss += current_batch_loss.item()
                valid_batches_for_loss_avg += 1
            
            if not (np.isnan(current_batch_accuracy) or np.isinf(current_batch_accuracy)):
                eval_total_accuracy += current_batch_accuracy
                valid_batches_for_acc_avg += 1
    
    avg_loss = eval_total_loss / valid_batches_for_loss_avg if valid_batches_for_loss_avg > 0 else 0.0
    avg_accuracy = eval_total_accuracy / valid_batches_for_acc_avg if valid_batches_for_acc_avg > 0 else 0.0
    
    return avg_loss, avg_accuracy

def plot_learning_curves(
    history_dict: Dict[str, List[float]],
    title_info_str: str,
    filename_path: Path
):
    """Plot and save learning curves for a training run."""
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['train_loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title(f'Loss: {title_info_str}', fontsize=9)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['train_acc'], label='Train BACC')
    plt.plot(history_dict['val_acc'], label='Val BACC')
    plt.title(f'BACC: {title_info_str}', fontsize=9)
    plt.xlabel('Epoch')
    plt.ylabel('BACC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename_path)
    plt.close()
    
    log_message(f"Saved learning curve: {filename_path.name}")

# --- Main Experiment Logic ---
def run_experiments():
    """Run the main experiment pipeline."""
    log_message(f"Experiment Script (Enhanced Parameters) started at {time.asctime(time.localtime())}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")
    log_message(f"Global Config: INPUT_DIM={INPUT_DIM}, NUM_CLASSES={NUM_CLASSES}, BATCH_SIZE={BATCH_SIZE}")
    log_message(f"Search Epochs: {NUM_EPOCHS_HYPERPARAM_SEARCH}, Final Train Epochs: {NUM_EPOCHS_FINAL_TRAIN}")

    # Create data splits
    train_file_ids, val_file_ids, test_file_ids = create_data_splits(
        FULL_FILE_LIST, 
        test_ratio=0.2, 
        val_ratio=0.1, 
        random_state=RANDOM_STATE
    )
    log_message(f"Data split: Train={len(train_file_ids)}, Val={len(val_file_ids)}, Test={len(test_file_ids)}")
    
    # Create datasets
    train_dataset = AudioFrameDataset(
        FEATURE_DIR, LABEL_DIR, train_file_ids, CLASS_NAMES, EXPECTED_FEATURE_KEYS
    )
    val_dataset = AudioFrameDataset(
        FEATURE_DIR, LABEL_DIR, val_file_ids, CLASS_NAMES, EXPECTED_FEATURE_KEYS
    )
    test_dataset = AudioFrameDataset(
        FEATURE_DIR, LABEL_DIR, test_file_ids, CLASS_NAMES, EXPECTED_FEATURE_KEYS
    )
    
    if not (len(train_dataset) and len(val_dataset) and len(test_dataset)):
        log_message("CRITICAL: Dataset empty. Exit.")
        return

    # Create dataloaders
    num_workers = 2 if device.type == 'cuda' and os.name != 'nt' else 0
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=BucketBatchSampler(train_dataset, BATCH_SIZE, shuffle=True),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_sampler=BucketBatchSampler(val_dataset, BATCH_SIZE, shuffle=False),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=BucketBatchSampler(test_dataset, BATCH_SIZE, shuffle=False),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )

    # Define loss function
    loss_function = nn.BCEWithLogitsLoss(reduction='none')
    
    # Define model configurations for hyperparameter search
    model_configurations: Dict[str, Dict[str, Any]] = {
        "MLP": {
            "model_class": SimpleTimeStepClassifier,
            "param_grid": {
                "hidden_dim": [128, 256, 512],
                "lr": [1e-3, 5e-4, 2e-4]
            }
        },
        "LSTM": {
            "model_class": LSTMClassifier,
            "param_grid": {
                "hidden_dim": [128, 256],
                "num_layers": [1, 2],
                "dropout_rate": [0.25, 0.4],
                "lr": [1e-3, 5e-4, 2e-4]
            }
        },
        "CNN1D": {
            "model_class": CNN1DClassifier,
            "param_grid": {
                "cnn_filters_1": [64, 128],
                "cnn_filters_2": [128, 256],
                "kernel_size": [3, 5],
                "dropout_rate": [0.25, 0.4],
                "lr": [1e-3, 5e-4, 2e-4]
            }
        }
    }
    
    # Track overall best results
    overall_best_test_bacc = -1.0
    overall_best_model_summary = {}
    all_family_results_list = []

    # Iterate through model architectures
    for model_name_str, model_config_dict in model_configurations.items():
        log_message(f"\n--- Experimenting with: {model_name_str} ---")
        best_val_bacc_for_family = -1.0
        best_params_for_family = None
        
        # Generate hyperparameter combinations
        param_names = list(model_config_dict["param_grid"].keys())
        param_value_combinations = list(itertools.product(
            *[model_config_dict["param_grid"][name] for name in param_names]
        ))

        # Try each hyperparameter combination
        for i, param_values_tuple in enumerate(param_value_combinations):
            current_run_hyperparams = {
                name: val for name, val in zip(param_names, param_values_tuple)
            }
            learning_rate = current_run_hyperparams.pop('lr')
            
            # Create descriptive strings for logging and filenames
            param_details_for_log = ", ".join(f"{k}={v}" for k, v in current_run_hyperparams.items()) + f", lr={learning_rate}"
            param_config_id_str = f"{model_name_str} Cfg{i+1}/{len(param_value_combinations)}"
            param_str_for_filename = "_".join([
                f"{k}_{v}" for k, v in current_run_hyperparams.items()
            ]).replace(".", "p") + f"_lr_{str(learning_rate).replace('.', 'p')}"
            
            log_message(f"\n{param_config_id_str}: {param_details_for_log}")

            # Initialize model, optimizer and scheduler
            model_instance = model_config_dict["model_class"](
                input_dim=INPUT_DIM,
                num_classes=NUM_CLASSES,
                **current_run_hyperparams
            ).to(device)
            
            optimizer_instance = optim.Adam(model_instance.parameters(), lr=learning_rate)
            scheduler_instance = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_instance,
                'max',
                factor=0.2,
                patience=max(2, NUM_EPOCHS_HYPERPARAM_SEARCH // 4),
                min_lr=1e-6
            )
            
            # Training loop for this hyperparameter configuration
            epoch_history_dict = {
                'train_loss': [], 
                'train_acc': [], 
                'val_loss': [], 
                'val_acc': []
            }
            current_config_best_val_bacc = -1.0
            
            for epoch_num in range(NUM_EPOCHS_HYPERPARAM_SEARCH):
                # Train and evaluate for one epoch
                train_loss, train_bacc = train_epoch_fn(
                    model_instance, train_dataloader, optimizer_instance, loss_function, device
                )
                
                val_loss, val_bacc = evaluate_model_fn(
                    model_instance, val_dataloader, loss_function, device
                )
                
                scheduler_instance.step(val_bacc)
                
                # Update history
                for key, val_list_item in zip(
                    epoch_history_dict.keys(),
                    [train_loss, train_bacc, val_loss, val_bacc]
                ):
                    epoch_history_dict[key].append(val_list_item)
                
                # Log progress
                log_message(
                    f"Ep{epoch_num+1} | "
                    f"Tr L:{train_loss:.3f} BACC:{train_bacc:.3f} | "
                    f"Val L:{val_loss:.3f} BACC:{val_bacc:.3f} | "
                    f"LR:{optimizer_instance.param_groups[0]['lr']:.1e}"
                )
                
                if val_bacc > current_config_best_val_bacc:
                    current_config_best_val_bacc = val_bacc
            
            # Update best params for this model family if better than previous
            if current_config_best_val_bacc > best_val_bacc_for_family:
                best_val_bacc_for_family = current_config_best_val_bacc
                best_params_for_family = {**current_run_hyperparams, 'lr': learning_rate}
            
            # Plot learning curves for this hyperparameter configuration
            plot_title_str = (
                f"{model_name_str}\n"
                f"{param_details_for_log}\n"
                f"Best Val BACC: {current_config_best_val_bacc:.3f}"
            )
            
            plot_filename_path = OUTPUT_DIR / (
                f"{model_name_str}_{param_str_for_filename}_"
                f"valBACC_{current_config_best_val_bacc:.3f}_curve.png"
            )
            
            plot_learning_curves(epoch_history_dict, plot_title_str, plot_filename_path)

        # Train final model with best hyperparameters
        if best_params_for_family:
            log_message(
                f"\nBest Params {model_name_str}: "
                f"{best_params_for_family} (Val BACC: {best_val_bacc_for_family:.4f})"
            )
            log_message(f"Retraining best {model_name_str} for {NUM_EPOCHS_FINAL_TRAIN} epochs...")
            
            # Extract final model parameters
            final_model_constructor_params = {
                k: v for k, v in best_params_for_family.items() if k != 'lr'
            }
            final_learning_rate = best_params_for_family['lr']
            
            # Initialize final model
            final_model_instance = model_config_dict["model_class"](
                input_dim=INPUT_DIM,
                num_classes=NUM_CLASSES,
                **final_model_constructor_params
            ).to(device)
            
            final_optimizer = optim.Adam(final_model_instance.parameters(), lr=final_learning_rate)
            final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                final_optimizer,
                'max',
                factor=0.2,
                patience=max(3, NUM_EPOCHS_FINAL_TRAIN // 5),
                min_lr=1e-6
            )
            
            # Training loop for final model
            final_epoch_history_dict = {
                'train_loss': [], 
                'train_acc': [], 
                'val_loss': [], 
                'val_acc': []
            }
            best_val_bacc_final_training = -1.0
            best_model_state_final_training = None
            
            for epoch_num in range(NUM_EPOCHS_FINAL_TRAIN):
                # Train and evaluate for one epoch
                train_loss, train_bacc = train_epoch_fn(
                    final_model_instance, train_dataloader, final_optimizer, loss_function, device
                )
                
                val_loss, val_bacc = evaluate_model_fn(
                    final_model_instance, val_dataloader, loss_function, device
                )
                
                final_scheduler.step(val_bacc)
                
                # Update history
                for key, val_list_item in zip(
                    final_epoch_history_dict.keys(),
                    [train_loss, train_bacc, val_loss, val_bacc]
                ):
                    final_epoch_history_dict[key].append(val_list_item)
                
                # Log progress
                log_message(
                    f"Final Ep{epoch_num+1} | "
                    f"Tr L:{train_loss:.3f} BACC:{train_bacc:.3f} | "
                    f"Val L:{val_loss:.3f} BACC:{val_bacc:.3f} | "
                    f"LR:{final_optimizer.param_groups[0]['lr']:.1e}"
                )
                
                # Save best model state
                if val_bacc > best_val_bacc_final_training:
                    best_val_bacc_final_training = val_bacc
                    best_model_state_final_training = final_model_instance.state_dict()
            
            # Plot final learning curves
            final_params_details_str = ", ".join(
                f"{k}={v}" for k, v in final_model_constructor_params.items()
            ) + f", lr={final_learning_rate}"
            
            plot_title_final_str = (
                f"Final: {model_name_str}\n"
                f"{final_params_details_str}\n"
                f"Val BACC: {best_val_bacc_final_training:.3f}"
            )
            
            plot_file_final_path = OUTPUT_DIR / (
                f"{model_name_str}_FINAL_valBACC_{best_val_bacc_final_training:.3f}_curve.png"
            )
            
            plot_learning_curves(final_epoch_history_dict, plot_title_final_str, plot_file_final_path)

            # Evaluate on test set and save model
            if best_model_state_final_training:
                final_model_instance.load_state_dict(best_model_state_final_training)
                test_loss, test_bacc = evaluate_model_fn(
                    final_model_instance, test_dataloader, loss_function, device
                )
                
                log_message(
                    f"**Final Performance {model_name_str} Test Set:** "
                    f"Loss: {test_loss:.4f}, BACC: {test_bacc:.4f}"
                )
                
                # Save model checkpoint
                checkpoint_filename = f"{model_name_str}_testBACC_{test_bacc:.4f}.pth"
                torch.save(
                    final_model_instance.state_dict(), 
                    OUTPUT_DIR / checkpoint_filename
                )
                log_message(f"Saved checkpoint: {checkpoint_filename}")
                
                # Record results
                family_summary_dict = {
                    "model_family": model_name_str,
                    "best_params": best_params_for_family,
                    "final_val_bacc": best_val_bacc_final_training,
                    "test_bacc": test_bacc,
                    "test_loss": test_loss,
                    "checkpoint": checkpoint_filename
                }
                
                all_family_results_list.append(family_summary_dict)
                
                # Track overall best model
                if test_bacc > overall_best_test_bacc:
                    overall_best_test_bacc = test_bacc
                    overall_best_model_summary = family_summary_dict
            else:
                log_message(f"No best model state from final training for {model_name_str}.")
        else:
            log_message(f"No successful hyperparameter config for {model_name_str}.")
    
    # Print overall summary
    log_message("\n--- Overall Experiment Summary (Sorted by Test BACC) ---")
    for result_dict in sorted(
        all_family_results_list,
        key=lambda x: x.get('test_bacc', -1.0),
        reverse=True
    ):
        log_message(
            f"  Family: {result_dict['model_family']}, "
            f"Params: {result_dict['best_params']}, "
            f"Test BACC: {result_dict.get('test_bacc', float('nan')):.4f}, "
            f"Ckpt: {result_dict['checkpoint']}"
        )
    
    # Print best model details
    if overall_best_model_summary:
        log_message(f"\n--- Overall Best Model ---")
        for k, v in overall_best_model_summary.items():
            formatted_value = v if not isinstance(v, float) else f'{v:.4f}'
            log_message(f"  {k.replace('_', ' ').title()}: {formatted_value}")
    else:
        log_message("No models qualified for overall best.")
    
    # Finish up
    log_message(f"Script finished at {time.asctime(time.localtime())}")
    log_message(f"Outputs in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    set_seeds(RANDOM_STATE)
    if initialize_global_dataset_vars():
        run_experiments()
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
              f"Exiting: Critical errors during global dataset variable initialization.")