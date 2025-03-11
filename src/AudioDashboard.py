import os, streamlit as st
from workers.DataExplorationWorker import DataExplorationWorker
from workers.ModelTrainingWorker import ModelTrainingWorker
from workers.ChallengeWorker import ChallengeWorker

def validate_path(path: str, path_type: str = "file") -> bool:
    if not path:
        return False
    if path_type == "file":
        return os.path.isfile(path)
    if path_type == "directory":
        return os.path.isdir(path)
    return False

def input_with_validation(label: str, default_value: str = "", path_type: str = "file", help_text: str = None) -> str:
    path = st.text_input(label, default_value, help=help_text)
    if path:
        if validate_path(path, path_type):
            st.markdown("✅ **Valid Path**")
        else:
            st.markdown("⚠️ **Invalid Path**")
    return path

def execute_worker(create_worker):
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        worker = create_worker(
            lambda p: progress_bar.progress(int(p)),
            lambda m: status_text.text(m)
        )
        status_text.text("🚀 Started!")
        result = worker.run()
        if result:
            progress_bar.progress(100)
            status_text.text("🎉 Completed!")
            st.balloons()
    except Exception as e:
        status_text.text(f"⚠️ Error: {e}")

def data_exploration_tab():
    st.subheader("Data Exploration")
    st.markdown("""
    **Goal**: Explore the annotated dataset, investigate clustering, and examine 
    correlations with textual embeddings.
    """)
    features_file = input_with_validation(
        label="Features & Embeddings File:",
        default_value="data/features_embeddings.csv",
        path_type="file",
        help_text="Path to a file containing precomputed features & text embeddings."
    )
    clustering_method = st.selectbox("Clustering Method:", ["KMeans", "DBSCAN", "Agglomerative"])
    n_clusters = st.slider("Number of Clusters", 2, 20, 5)
    embedding_type = st.selectbox("Text Embedding Type:", ["Word2Vec", "BERT", "SentenceTransformer", "Custom"])
    if st.button("Run Data Exploration"):
        if not validate_path(features_file, "file"):
            st.error("Invalid features file path.")
            return
        def create_worker(progress_cb, status_cb):
            return DataExplorationWorker(
                features_file=features_file,
                clustering_method=clustering_method,
                n_clusters=n_clusters,
                embedding_type=embedding_type,
                progress_callback=progress_cb,
                status_callback=status_cb
            )
        execute_worker(create_worker)

def model_training_tab():
    st.subheader("Training a Model Based on Inferred Labels")
    st.markdown("""
    **Goal**: Train a sound event detection model, using the textual annotations 
    mapped to the desired labels.
    """)
    training_data_file = input_with_validation(
        label="Training Data File:",
        default_value="data/training_data.csv",
        path_type="file",
        help_text="Path to CSV with textual descriptors mapped to training labels."
    )
    label_mapping_mode = st.radio("Label Mapping Strategy:", ["Simple Synonyms", "Embedding Similarity"])
    synonyms_list = st.text_area("Synonym List (comma or newline separated):",
        "speech,talking,conversation\ncar,vehicle,honk\nmusic,song,tune")
    embedding_model = st.selectbox("Embedding Model for Label Mapping:", ["Word2Vec", "BERT", "SentenceTransformer"])
    model_type = st.selectbox("Model Type:", ["RandomForest", "SVM", "CNN", "RNN", "Transformer"])
    st.write("### Hyperparameters")
    lr = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001)
    epochs = st.slider("Epochs", 1, 200, 10)
    batch_size = st.slider("Batch Size", 1, 256, 32)
    if st.button("Start Model Training"):
        if not validate_path(training_data_file, "file"):
            st.error("Invalid training data file.")
            return
        hyperparams = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size
        }
        def create_worker(progress_cb, status_cb):
            return ModelTrainingWorker(
                training_data_file=training_data_file,
                label_mapping_mode=label_mapping_mode,
                synonyms_list=synonyms_list,
                embedding_model=embedding_model,
                model_type=model_type,
                hyperparams=hyperparams,
                progress_callback=progress_cb,
                status_callback=status_cb
            )
        execute_worker(create_worker)

def challenge_tab():
    st.subheader("Challenge")
    st.markdown("""
    **Goal**: Generate predictions on a secret test set to get an unbiased measure 
    of the model's performance.
    """)
    trained_model_path = input_with_validation(
        label="Trained Model Path:",
        default_value="models/best_model.pth",
        path_type="file",
        help_text="Path to the trained model file."
    )
    secret_test_set_path = input_with_validation(
        label="Secret Test Set Directory:",
        default_value="data/secret_test/",
        path_type="directory",
        help_text="Directory containing hidden or final test audio."
    )
    if st.button("Generate Predictions"):
        if not validate_path(trained_model_path, "file"):
            st.error("Invalid model path.")
            return
        if not validate_path(secret_test_set_path, "directory"):
            st.error("Invalid test set directory.")
            return
        def create_worker(progress_cb, status_cb):
            return ChallengeWorker(
                trained_model_path=trained_model_path,
                secret_test_set_path=secret_test_set_path,
                progress_callback=progress_cb,
                status_callback=status_cb
            )
        execute_worker(create_worker)

def main():
    st.set_page_config(page_title="MLPC Project Dashboard", page_icon="🔊", layout="wide")
    st.title("🔊 MLPC Project — Audio SED Dashboard")
    tabs = st.tabs(["Data Exploration", "Model Training", "Challenge"])
    with tabs[0]:
        data_exploration_tab()
    with tabs[1]:
        model_training_tab()
    with tabs[2]:
        challenge_tab()

if __name__ == "__main__":
    main()