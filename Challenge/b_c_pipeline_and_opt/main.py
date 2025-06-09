import os, math, time, warnings, torch, torch.nn as nn, numpy as np, pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import List, Dict, Any, Callable
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DEV_DATASET_PATH = BASE_DIR / "MLPC2025_classification"
    TEST_DATASET_PATH = BASE_DIR / "MLPC2025_test"
    MODEL_CHECKPOINT_PATH = BASE_DIR / "CNN1D_testBACC_0.8201.pth"
    OUTPUT_DIR = BASE_DIR / "submission_outputs"
    
    TARGET_CLASSES = ['Speech', 'Shout', 'Chainsaw', 'Jackhammer', 'Lawn Mower', 'Power Drill', 'Dog Bark', 'Rooster Crow', 'Horn Honk', 'Siren']
    SMOOTHING_CLASSES = ['Shout', 'Chainsaw', 'Jackhammer', 'Lawn Mower', 'Power Drill', 'Horn Honk', 'Siren']
    
    RANDOM_STATE = 42
    OPT_ITERATIONS = 5 
    
    ALL_ORIGINAL_CLASSES: List[str] = []
    EXPECTED_FEATURE_KEYS: List[str] = []
    INPUT_DIM: int = 0

    @staticmethod
    def setup_dynamic_config():
        print("--- Dynamically configuring vocabulary and features from dataset ---")
        label_dir = Config.DEV_DATASET_PATH / "labels"
        assert label_dir.is_dir(), f"Label directory not found at {label_dir}"
        all_keys = set()
        for label_file in tqdm(list(label_dir.glob("*.npz")), desc="Discovering vocabulary", leave=False, ncols=100):
            with np.load(label_file) as data:
                all_keys.update(data.keys())
        Config.ALL_ORIGINAL_CLASSES = sorted(list(all_keys))
        
        feature_dir = Config.DEV_DATASET_PATH / "audio_features"
        assert feature_dir.is_dir(), f"Feature directory not found at {feature_dir}"
        sample_feature_file = next(feature_dir.glob("*.npz"))
        with np.load(sample_feature_file) as data:
            Config.EXPECTED_FEATURE_KEYS = list(data.keys())
            Config.INPUT_DIM = sum(data[k].shape[1] if data[k].ndim > 1 else 1 for k in Config.EXPECTED_FEATURE_KEYS)
        
        print(f"Discovered {len(Config.ALL_ORIGINAL_CLASSES)} classes in total.")
        print(f"Discovered {len(Config.EXPECTED_FEATURE_KEYS)} feature types, with total input dim {Config.INPUT_DIM}.")

    @staticmethod
    def verify():
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        for path in [Config.DEV_DATASET_PATH, Config.TEST_DATASET_PATH, Config.MODEL_CHECKPOINT_PATH]:
            assert path.exists(), f"Required path not found: {path}"
        print("Configuration and paths verified.")

COST_MATRIX = {"Speech":{"FP":1,"FN":5},"Dog Bark":{"FP":1,"FN":5},"Rooster Crow":{"FP":1,"FN":5},"Shout":{"FP":2,"FN":10},"Lawn Mower":{"FP":3,"FN":15},"Chainsaw":{"FP":3,"FN":15},"Jackhammer":{"FP":3,"FN":15},"Power Drill":{"FP":3,"FN":15},"Horn Honk":{"FP":3,"FN":15},"Siren":{"FP":3,"FN":15}}

class CNN1DClassifier(nn.Module):
    def __init__(self, i, o, c1, c2, k, d):
        super().__init__(); self.conv_block1=nn.Sequential(nn.Conv1d(i,c1,k,padding='same'), nn.BatchNorm1d(c1), nn.ReLU(), nn.Dropout(d)); self.conv_block2=nn.Sequential(nn.Conv1d(c1,c2,k,padding='same'), nn.BatchNorm1d(c2), nn.ReLU(), nn.Dropout(d)); self.output_conv=nn.Conv1d(c2,o,1)
    def forward(self, x): x=x.permute(0,2,1);x=self.conv_block1(x);x=self.conv_block2(x);x=self.output_conv(x);return x.permute(0,2,1)

def calculate_total_cost(pred_df, gt_df):
    pred_aligned, gt_aligned=pred_df.set_index(['filename','onset']), gt_df.set_index(['filename','onset'])
    common_index = pred_aligned.index.intersection(gt_aligned.index)
    pred_aligned, gt_aligned = pred_aligned.loc[common_index], gt_aligned.loc[common_index]
    total_cost = 0
    for cls in Config.TARGET_CLASSES:
        fp_rate = ((pred_aligned[cls]==1)&(gt_aligned[cls]==0)).mean()
        fn_rate = ((pred_aligned[cls]==0)&(gt_aligned[cls]==1)).mean()
        total_cost += (COST_MATRIX[cls]["FP"] * fp_rate + COST_MATRIX[cls]["FN"] * fn_rate) * 50
    return total_cost

def max_agg(arr): return np.max(arr, axis=1)
def mean_agg(arr): return np.mean(arr, axis=1)
def hybrid_agg(arr): return 0.7 * np.max(arr, axis=1) + 0.3 * np.mean(arr, axis=1)

def aggregate_frames_to_segments(arr, agg_func: Callable, factor=10):
    n_f, n_d = arr.shape; n_s = math.ceil(n_f / factor)
    p_arr = np.zeros((n_s * factor, n_d), dtype=arr.dtype); p_arr[:n_f] = arr
    reshaped_arr = p_arr.reshape(n_s, factor, n_d)
    return agg_func(reshaped_arr)

def apply_temporal_smoothing(df: pd.DataFrame, classes_to_smooth: List[str]) -> pd.DataFrame:
    df_out = df.copy()
    for class_name in classes_to_smooth:
        grouped = df_out.groupby('filename')[class_name]
        prev_val = grouped.shift(1)
        next_val = grouped.shift(-1)
        condition = (df_out[class_name] == 0) & (prev_val == 1) & (next_val == 1)
        df_out.loc[condition, class_name] = 1
    return df_out

def get_ground_truth_df(filenames, dataset_path):
    rows=[]
    for fname in tqdm(filenames, desc="Processing ground truth", leave=False, ncols=100):
        base=os.path.splitext(fname)[0]
        try:
            labels_npz=np.load(dataset_path/'labels'/f"{base}_labels.npz")
            n_steps=np.load(dataset_path/'audio_features'/f"{base}.npz")['embeddings'].shape[0]
            class_arrays=[labels_npz.get(cls, np.zeros((n_steps, 1))).mean(-1) for cls in Config.TARGET_CLASSES]
            agg_lbls=aggregate_frames_to_segments(np.stack(class_arrays, axis=1), max_agg)
            for i,row in enumerate(agg_lbls): rows.append([fname, round(i*1.2,1)] + (row>0).astype(int).tolist())
        except FileNotFoundError: warnings.warn(f"Feature file for {fname} not found. Skipping.")
    return pd.DataFrame(data=rows, columns=["filename", "onset"] + Config.TARGET_CLASSES)

def get_frame_level_predictions(file_ids, feature_dir, model, device):
    class PredictionDataset(Dataset):
        def __init__(self, f_ids, f_dir): self.f_dir, self.f_ids = f_dir, f_ids
        def __len__(self): return len(self.f_ids)
        def __getitem__(self, idx):
            fname=self.f_ids[idx]; fpath=self.f_dir/f"{os.path.splitext(fname)[0]}.npz"
            try:
                feats_npz=np.load(fpath); feat_arrs=[feats_npz[k] for k in Config.EXPECTED_FEATURE_KEYS]; min_len=min(a.shape[0] for a in feat_arrs)
                if min_len==0: return None,fname
                trimmed=[(a if a.ndim>1 else a[:,np.newaxis])[:min_len] for a in feat_arrs]
                return torch.from_numpy(np.concatenate(trimmed,axis=-1)).float(),fname
            except Exception as e: warnings.warn(f"Error loading {fname}: {e}"); return None,fname
    dataset=PredictionDataset(file_ids, feature_dir); predictions={}
    target_indices = [Config.ALL_ORIGINAL_CLASSES.index(cls) for cls in Config.TARGET_CLASSES]
    model.eval()
    with torch.no_grad():
        for features, filename in tqdm(dataset, desc=f"Predicting on {os.path.basename(feature_dir.parent)}", leave=False, ncols=100):
            if features is None: continue
            probs=torch.sigmoid(model(features.unsqueeze(0).to(device)))
            predictions[filename] = probs[:, :, target_indices].squeeze(0).cpu().numpy()
    return predictions

def run_coordinate_descent_optimization(probs_df, gt_df, post_processing_func=None):
    current_thresholds = {cls:0.5 for cls in Config.TARGET_CLASSES}
    temp_preds = probs_df.copy()
    for c in Config.TARGET_CLASSES: temp_preds[c] = (temp_preds[c] > 0.5).astype(int)
    
    if post_processing_func:
        temp_preds = post_processing_func(temp_preds)

    min_overall_cost = calculate_total_cost(temp_preds, gt_df)

    for i in range(Config.OPT_ITERATIONS):
        cost_at_start_of_iter = min_overall_cost
        for cls in tqdm(Config.TARGET_CLASSES, desc=f"Optimizing thresholds (iter {i+1}/{Config.OPT_ITERATIONS})", leave=False, ncols=100):
            best_t_for_cls, cost_at_best_t = current_thresholds[cls], min_overall_cost
            search_space = np.unique(np.quantile(probs_df[cls][probs_df[cls]>1e-4], np.linspace(0,1,150))) if (probs_df[cls]>1e-4).any() else [0.01]
            
            for t in search_space:
                temp_thresholds = current_thresholds.copy(); temp_thresholds[cls]=t
                temp_preds_df = probs_df.copy()
                for c, th in temp_thresholds.items(): temp_preds_df[c] = (temp_preds_df[c] > th).astype(int)
                
                if post_processing_func:
                    temp_preds_df = post_processing_func(temp_preds_df)
                
                cost = calculate_total_cost(temp_preds_df, gt_df)
                if cost < cost_at_best_t:
                    cost_at_best_t, best_t_for_cls = cost, t
            
            current_thresholds[cls], min_overall_cost = best_t_for_cls, cost_at_best_t
        
        if np.isclose(min_overall_cost, cost_at_start_of_iter, atol=1e-4):
            print(f"\nConvergence reached at iteration {i+1}. Final cost: {min_overall_cost:.4f}")
            break
            
    return {k:round(v,4) for k,v in current_thresholds.items()}, min_overall_cost

def apply_pipeline(frame_preds: Dict[str, np.ndarray], config: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    agg_func = config['agg_func']
    for filename, prob_matrix in frame_preds.items():
        segment_probs = aggregate_frames_to_segments(prob_matrix, agg_func)
        for i, p_row in enumerate(segment_probs): rows.append([filename, round(i * 1.2, 1)] + p_row.tolist())
    
    probs_df = pd.DataFrame(rows, columns=["filename","onset"]+Config.TARGET_CLASSES)
    
    final_df = probs_df.copy()
    for c in Config.TARGET_CLASSES:
        final_df[c] = (final_df[c] > config['thresholds'][c]).astype(int)
        
    if config.get('post_process', False):
        final_df = apply_temporal_smoothing(final_df, Config.SMOOTHING_CLASSES)
        
    return final_df

if __name__ == "__main__":
    start_time = time.time()
    results_log = ["="*80, "MLPC 2025 CHALLENGE: PIPELINE LOG", "="*80]
    Config.verify()
    Config.setup_dynamic_config()
    
    print("\n--- PHASE 1: Data Setup ---")
    metadata=pd.read_csv(Config.DEV_DATASET_PATH/'metadata.csv')
    all_files=metadata['filename'].unique()
    gt_all = get_ground_truth_df(all_files, Config.DEV_DATASET_PATH)
    file_labels = gt_all.groupby('filename')[Config.TARGET_CLASSES].max().reindex(all_files, fill_value=0)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.RANDOM_STATE)
    train_idx, val_idx = next(msss.split(file_labels.index, file_labels.values))
    train_files, val_files = file_labels.index[train_idx].tolist(), file_labels.index[val_idx].tolist()
    gt_val_df = gt_all[gt_all['filename'].isin(val_files)].copy()

    print("\n--- PHASE 2: Model Loading & Frame-Level Prediction ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DClassifier(Config.INPUT_DIM, len(Config.ALL_ORIGINAL_CLASSES), 128, 128, 5, 0.4).to(device)
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device, weights_only=True))
    print(f"Loaded model onto {device}.")
    val_frame_preds = get_frame_level_predictions(val_files, Config.DEV_DATASET_PATH/"audio_features", model, device)
    
    print("\n--- PHASE 3: Investigating Improvement Strategies ---")
    
    strategy_configs = [
        {'name': 'Max Aggregation', 'agg_func': max_agg, 'post_process': False},
        {'name': 'Mean Aggregation', 'agg_func': mean_agg, 'post_process': False},
        {'name': 'Hybrid Aggregation (0.7max+0.3mean)', 'agg_func': hybrid_agg, 'post_process': False},
        {'name': 'Max Agg + Temporal Smoothing', 'agg_func': max_agg, 'post_process': True},
    ]
    
    evaluated_pipelines = []

    for config in strategy_configs:
        print(f"\n-- Evaluating Strategy: {config['name']} --")
        
        rows = [[f, round(i*1.2,1)]+p.tolist() for f, p_mat in val_frame_preds.items() for i,p in enumerate(aggregate_frames_to_segments(p_mat, config['agg_func']))]
        probs_df = pd.DataFrame(rows, columns=["filename","onset"]+Config.TARGET_CLASSES)
        
        post_func = None
        if config['post_process']:
            post_func = lambda df: apply_temporal_smoothing(df, Config.SMOOTHING_CLASSES)
        
        thresholds, cost = run_coordinate_descent_optimization(probs_df, gt_val_df, post_func)
        
        pipeline_result = config.copy()
        pipeline_result.update({'thresholds': thresholds, 'cost': cost})
        evaluated_pipelines.append(pipeline_result)
        results_log.append(f"Strategy ({config['name']}): Final Validation Cost = {cost:.4f}")

    print("\n" + "="*80)
    print("--- CHAMPION PIPELINE SELECTION ---")
    
    champion_config = min(evaluated_pipelines, key=lambda p: p['cost'])
    
    results_log.extend(["\n"+"="*80, "--- CHAMPION PIPELINE SELECTION ---"])
    for p in sorted(evaluated_pipelines, key=lambda p: p['cost']):
        results_log.append(f"  - Strategy: {p['name']}, Cost: {p['cost']:.4f}")
    
    results_log.append(f"\n🏆 Winning Strategy: '{champion_config['name']}' with cost {champion_config['cost']:.4f}")
    results_log.append(f"   - Aggregation: {champion_config['agg_func'].__name__}")
    results_log.append(f"   - Post-processing Smoothing: {'Enabled' if champion_config['post_process'] else 'Disabled'}")
    results_log.append(f"   - Optimal Thresholds: {champion_config['thresholds']}")

    print(f"\n🏆 Champion pipeline selected: '{champion_config['name']}' with a validation cost of {champion_config['cost']:.4f}")
    
    print("\n--- Generating final submission file with champion pipeline ---")
    
    test_files_with_ext = [f.stem + ".mp3" for f in (Config.TEST_DATASET_PATH/"audio_features").glob("*.npz")]
    
    test_frame_preds = get_frame_level_predictions(test_files_with_ext, Config.TEST_DATASET_PATH/"audio_features", model, device)
    
    final_submission_df = apply_pipeline(test_frame_preds, champion_config)
    
    submission_path = Config.OUTPUT_DIR/"predictions.csv"
    final_submission_df.to_csv(submission_path, index=False)
    results_log.append(f"\nFinal submission file generated at: {submission_path}")
    
    summary_text = "\n".join(results_log)
    print("\n" + summary_text)
    (Config.OUTPUT_DIR/"run_summary_log.txt").write_text(summary_text, encoding='utf-8')
    print(f"\nFull summary log saved to: {Config.OUTPUT_DIR/'run_summary_log.txt'}")
    print(f"Total script execution time: {time.time()-start_time:.2f} seconds.")