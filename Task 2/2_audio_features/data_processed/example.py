import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from audio_features_utils import (
    load_audio_pipeline, load_processed_data,
    extract_region_features, get_combined_feature_vector,
    process_features, predict_cluster
)

# 1. Load pipeline & data
pipeline = load_audio_pipeline('../models/audio_features_pipeline.pkl')
proc_data = load_processed_data('processed_audio_features.pkl')

print("Pipeline components:")
for k in pipeline.keys():
    print(f"  - {k}")
print(f"Processed regions: {len(proc_data['regions_data'])}")

# 2. Load metadata & annotations
metadata_df = pd.read_csv('../metadata.csv')
annotations_df = pd.read_csv('../annotations.csv')

# 3. Process a sample annotation
sample = annotations_df.iloc[1000]
fn = sample['filename']
st, en = sample['onset'], sample['offset']
text = sample['text']

feats = extract_region_features(fn, st, en, '../audio_features', metadata_df)
vec = get_combined_feature_vector(feats).reshape(1,-1)

proc = process_features(vec, pipeline)
cluster = predict_cluster(vec, pipeline)[0]

print(f"Sample: '{text}' ({st:.2f}-{en:.2f}s) -> Cluster {cluster}")

# 4. Compare cluster members
members = [r['text'] for r in proc_data['regions_data'] if r.get('cluster')==cluster and not r['is_silent']]
print("Top in cluster:")
for t in pd.Series(members).value_counts().head(3).index:
    print(" -", t)

# 5. Quick t-SNE plot
coords = proc_data['features_tsne']
labels = np.array([r.get('cluster',-1) for r in proc_data['regions_data']])
plt.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', s=10)
plt.title('t-SNE')
plt.show()