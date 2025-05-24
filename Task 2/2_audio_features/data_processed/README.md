# Audio Features Analysis

## Available Resources

1. **Processed Data**: 
   - `processed_audio_features.pkl`: Contains processed feature vectors, clustering results, and analysis data

2. **Utility Code**:
   - `audio_features_utils.py`: A Python module with utility functions for working with audio features
   - `example.py`: An example script showing how to use the processed data and utilities

3. **Saved Models**:
   - `../models/audio_features_pipeline.pkl`: The feature processing pipeline (scaler, PCA, KMeans)

## How to Use These Resources

1. **For accessing processed data:**
   ```python
   import pickle

   # Load the processed data
   with open('processed_audio_features.pkl', 'rb') as f:
       data = pickle.load(f)

   # Access the data components
   regions_data = data['regions_data']  # Full region info including annotations and features
   features_tsne = data['features_tsne']  # t-SNE coordinates for visualization
   cluster_labels = data['cluster_labels']  # Cluster assignments
   ```

2. **For processing new audio regions:**
   ```python
   from audio_features_utils import extract_region_features, get_combined_feature_vector, process_features

   # Extract features for a new region
   region_features = extract_region_features(filename, start_time, end_time, audio_features_path, metadata_df)
   feature_vector = get_combined_feature_vector(region_features)

   # Load the pipeline and process the features
   with open('../models/audio_features_pipeline.pkl', 'rb') as f:
       pipeline = pickle.load(f)

   processed_features = process_features(feature_vector.reshape(1, -1), pipeline)
   ```

3. **For cluster prediction:**
   ```python
   from audio_features_utils import predict_cluster

   # Predict cluster for a new feature vector
   cluster = predict_cluster(feature_vector.reshape(1, -1), pipeline)[0]
   print(f"Assigned to cluster: {cluster}")
   ```

See `example.py` for a complete example of how to use these resources.