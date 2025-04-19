"""
Audio Features Utilities Module

This module provides functions for working with audio features extracted from audio files,
including loading, processing, feature extraction from specific regions, and identification
of silent regions in audio files.
"""

import os
import pickle
import numpy as np


def get_feature_names(audio_features_path, metadata_df):
    """
    Generate feature names based on the available audio features.
    
    Parameters:
        audio_features_path (str): Path to the directory containing audio feature files
        metadata_df (DataFrame): DataFrame containing metadata with a 'filename' column
        
    Returns:
        list: List of feature names in the format {feature_type}_{index}
    """
    sample_filename = metadata_df.loc[0, "filename"].replace("mp3", "npz")
    features = np.load(os.path.join(audio_features_path, sample_filename))

    feature_names = []
    for feature_type in sorted(features.keys()):
        n = features[feature_type].shape[1]
        feature_names.extend([f"{feature_type}_{i}" for i in range(n)])
    return feature_names


def extract_region_features(filename, start_time, end_time, features_path, metadata_df, frame_rate=8.33):
    """
    Extract features from a specific time region in an audio file.
    
    Parameters:
        filename (str): Name of the audio file
        start_time (float): Start time of the region in seconds
        end_time (float): End time of the region in seconds
        features_path (str): Path to the directory containing feature files
        metadata_df (DataFrame): DataFrame containing audio metadata
        frame_rate (float): Frame rate of the feature extraction in frames per second (default: 8.33)
        
    Returns:
        dict: Dictionary of average feature values for the specified region, or None if features cannot be loaded
    """
    try:
        npz_name = filename.replace("mp3", "npz")
        features = np.load(os.path.join(features_path, npz_name))
    except Exception as e:
        print(f"Error loading features for {filename}: {e}")
        return None

    start_frame = max(0, int(start_time * frame_rate))
    end_frame = min(int(end_time * frame_rate), features[list(features.keys())[0]].shape[0] - 1)
    if end_frame <= start_frame:
        end_frame = start_frame + 1

    region = {}
    for ft, arr in features.items():
        if start_frame >= arr.shape[0]:
            region[ft] = arr[-1]
        else:
            region[ft] = np.mean(arr[start_frame:end_frame+1], axis=0)
    return region


def get_combined_feature_vector(region_features):
    """
    Combine different feature types into a single feature vector.
    
    Parameters:
        region_features (dict): Dictionary of features extracted from an audio region
        
    Returns:
        numpy.ndarray: Combined feature vector, or None if region_features is None
    """
    if region_features is None:
        return None
    combined = []
    for ft in sorted(region_features):
        combined.extend(region_features[ft])
    return np.array(combined)


def identify_silent_regions(filename, annotations_df, metadata_df, features_path, min_duration=0.5):
    """
    Identify silent regions in an audio file based on annotation gaps.
    
    Parameters:
        filename (str): Name of the audio file
        annotations_df (DataFrame): DataFrame containing audio annotations
        metadata_df (DataFrame): DataFrame containing audio metadata
        features_path (str): Path to the directory containing feature files
        min_duration (float): Minimum duration in seconds for a region to be considered silent (default: 0.5)
        
    Returns:
        list: List of tuples (start_time, end_time) for silent regions
    """
    ann = annotations_df[annotations_df['filename'] == filename].sort_values('onset')
    if ann.empty:
        return []

    try:
        npz_name = filename.replace("mp3", "npz")
        feats = np.load(os.path.join(features_path, npz_name))
        total = feats[list(feats.keys())[0]].shape[0] / 8.33
    except Exception:
        return []

    silent = []
    if ann.iloc[0]['onset'] > min_duration:
        silent.append((0, ann.iloc[0]['onset']))
    for i in range(len(ann)-1):
        end_i = ann.iloc[i]['offset']
        start_j = ann.iloc[i+1]['onset']
        if start_j - end_i > min_duration:
            silent.append((end_i, start_j))
    if total - ann.iloc[-1]['offset'] > min_duration:
        silent.append((ann.iloc[-1]['offset'], total))
    return silent


def extract_features_batch(annotations_df, metadata_df, audio_features_path, max_annotations=1000, n_silent_files=100):
    """
    Extract features from a batch of audio annotations and silent regions.
    
    Parameters:
        annotations_df (DataFrame): DataFrame containing audio annotations
        metadata_df (DataFrame): DataFrame containing audio metadata
        audio_features_path (str): Path to the directory containing feature files
        max_annotations (int): Maximum number of annotations to process (default: 1000)
        n_silent_files (int): Number of files to sample for silent regions (default: 100)
        
    Returns:
        list: List of dictionaries containing extracted features and metadata for each region
    """
    np.random.seed(42)
    sample = annotations_df.sample(min(len(annotations_df), max_annotations))
    results = []

    for i, row in sample.iterrows():
        feats = extract_region_features(
            row['filename'], row['onset'], row['offset'],
            audio_features_path, metadata_df
        )
        if feats is None:
            continue
        results.append({
            'filename': row['filename'],
            'onset': row['onset'],
            'offset': row['offset'],
            'duration': row['offset']-row['onset'],
            'text': row['text'],
            'is_silent': False,
            'features': get_combined_feature_vector(feats)
        })

    files = np.random.choice(metadata_df['filename'], min(n_silent_files, len(metadata_df)), replace=False)
    for fn in files:
        silent = identify_silent_regions(fn, annotations_df, metadata_df, audio_features_path)
        for j, (s,e) in enumerate(silent):
            feats = extract_region_features(fn, s, e, audio_features_path, metadata_df)
            if feats is None:
                continue
            results.append({
                'filename': fn,
                'onset': s,
                'offset': e,
                'duration': e-s,
                'text': 'silent_region',
                'is_silent': True,
                'features': get_combined_feature_vector(feats)
            })
    print(f"Extracted features for {len(results)} regions")
    return results


def load_audio_pipeline(path='../models/audio_features_pipeline.pkl'):
    """
    Load a pre-trained audio processing pipeline from disk.
    
    Parameters:
        path (str): Path to the saved pipeline file (default: '../models/audio_features_pipeline.pkl')
        
    Returns:
        dict: Dictionary containing the loaded pipeline components, or None if loading fails
    """
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return None


def process_features(features, pipeline):
    """
    Process audio features using a pre-trained pipeline.
    
    Parameters:
        features (numpy.ndarray): Audio features to process
        pipeline (dict): Dictionary containing 'scaler' and 'pca' components
        
    Returns:
        numpy.ndarray: Processed features after scaling and PCA transformation
    """
    return pipeline['pca'].transform(pipeline['scaler'].transform(features))


def predict_cluster(features, pipeline):
    """
    Predict the cluster of audio features using a pre-trained pipeline.
    
    Parameters:
        features (numpy.ndarray): Audio features to cluster
        pipeline (dict): Dictionary containing 'kmeans', 'scaler', and 'pca' components
        
    Returns:
        numpy.ndarray: Cluster assignments for the input features
    """
    return pipeline['kmeans'].predict(process_features(features, pipeline))


def save_processed_data(data, output_dir='data_processed', filename='processed_audio_features.pkl'):
    """
    Save processed audio feature data to disk.
    
    Parameters:
        data: Data to save
        output_dir (str): Directory to save the data (default: 'data_processed')
        filename (str): Name of the output file (default: 'processed_audio_features.pkl')
        
    Returns:
        str: Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to '{out}'")
    return out


def load_processed_data(path):
    """
    Load processed audio feature data from disk.
    
    Parameters:
        path (str): Path to the saved data file
        
    Returns:
        object: Loaded data, or None if loading fails
    """
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None