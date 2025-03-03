import os
import streamlit as st
import DataPreparationWorker, TrainingWorker, EvaluationWorker

st.set_page_config(
    page_title="Audio ML Management Dashboard",
    page_icon="🔊",
    layout="wide"
)

def validate_path(path: str, path_type: str = "file") -> bool:
    """
    Validates that a path exists and is of the correct type (file or directory).
    """
    if not path:
        return False
    if path_type == "file":
        return os.path.isfile(path)
    if path_type == "directory":
        return os.path.isdir(path)
    return False

def input_with_validation(label: str, default_value: str = "", path_type: str = "file", help_text: str = None) -> str:
    """
    Renders a text input in Streamlit for a file or directory path and checks its validity.
    A small message is displayed indicating whether the path is valid.
    """
    path = st.text_input(label, default_value, help=help_text)
    if path:
        if validate_path(path, path_type):
            st.markdown("✅ **Valid Path**")
        else:
            st.markdown("⚠️ **Invalid Path**")
    return path

def execute_worker(create_worker):
    """
    A helper function that:
      1. Initializes a worker with progress/status callbacks.
      2. Runs the worker.
      3. Updates the UI with progress and final status.
    """
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
        else:
            status_text.text("⚠️ Failed.")
    except Exception as e:
        status_text.text(f"⚠️ Error: {e}")

def data_preparation_tab():
    """
    Handles data preparation steps for raw audio files.
    """
    st.subheader("Data Preparation")

    raw_audio_dir = input_with_validation(
        "Raw Audio Directory:",
        "data/raw_audio",
        path_type="directory",
        help_text="Folder containing raw audio files."
    )
    annotations_file = input_with_validation(
        "Annotations File:",
        "data/annotations.csv",
        path_type="file",
        help_text="File with labels or other relevant metadata."
    )
    processed_data_dir = input_with_validation(
        "Processed Data Output Directory:",
        "data/processed_audio",
        path_type="directory",
        help_text="Destination for processed audio files."
    )
    sample_rate = st.number_input("Sample Rate (Hz):", min_value=8000, max_value=48000, value=16000, step=1000)

    if st.button("Start Data Preparation"):
        # Basic validation of paths
        if not validate_path(raw_audio_dir, "directory"):
            st.error("Invalid raw audio directory.")
            return
        if not validate_path(annotations_file, "file"):
            st.error("Invalid annotations file.")
            return
        if not validate_path(processed_data_dir, "directory"):
            st.error("Invalid output directory for processed data.")
            return

        execute_worker(lambda progress_cb, status_cb: DataPreparationWorker(
            raw_audio_dir=raw_audio_dir,
            annotations_file=annotations_file,
            processed_data_dir=processed_data_dir,
            sample_rate=sample_rate,
            progress_callback=progress_cb,
            status_callback=status_cb
        ))

def training_tab():
    """
    Handles the training configuration for an audio model.
    """
    st.subheader("Training")

    epochs = st.number_input("Epochs:", min_value=1, max_value=1000, value=10)
    batch_size = st.number_input("Batch Size:", min_value=1, max_value=2048, value=32)
    learning_rate = st.number_input("Learning Rate:", min_value=1e-6, max_value=1.0, value=0.001, format="%.6f")

    train_data_path = input_with_validation(
        "Training Data File:",
        "data/processed_audio/train_data.csv",
        path_type="file"
    )
    val_data_path = input_with_validation(
        "Validation Data File (optional):",
        "data/processed_audio/val_data.csv",
        path_type="file"
    )
    model_output_dir = input_with_validation(
        "Model Output Directory:",
        "models/",
        path_type="directory"
    )

    if st.button("Start Training"):
        if not validate_path(train_data_path, "file"):
            st.error("Invalid training data file.")
            return
        if val_data_path and not validate_path(val_data_path, "file"):
            st.error("Invalid validation data file.")
            return
        if not validate_path(model_output_dir, "directory"):
            st.error("Invalid model output directory.")
            return

        execute_worker(lambda progress_cb, status_cb: TrainingWorker(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_data_path=train_data_path,
            val_data_path=val_data_path or None,
            model_output_dir=model_output_dir,
            progress_callback=progress_cb,
            status_callback=status_cb
        ))

def evaluation_tab():
    """
    Handles model evaluation using a test dataset.
    """
    st.subheader("Evaluation")

    model_path = input_with_validation(
        "Trained Model Path:",
        "models/audio_model.pth",
        path_type="file"
    )
    test_data_path = input_with_validation(
        "Test Data File:",
        "data/processed_audio/test_data.csv",
        path_type="file"
    )
    results_output_dir = input_with_validation(
        "Evaluation Results Directory:",
        "evaluation/",
        path_type="directory"
    )

    if st.button("Start Evaluation"):
        if not validate_path(model_path, "file"):
            st.error("Invalid model file.")
            return
        if not validate_path(test_data_path, "file"):
            st.error("Invalid test data file.")
            return
        if not validate_path(results_output_dir, "directory"):
            st.error("Invalid results directory.")
            return

        execute_worker(lambda progress_cb, status_cb: EvaluationWorker(
            model_path=model_path,
            test_data_path=test_data_path,
            results_output_dir=results_output_dir,
            progress_callback=progress_cb,
            status_callback=status_cb
        ))

def main():
    st.title("Audio ML Management Dashboard")
    tabs = st.tabs(["Data Preparation", "Training", "Evaluation"])

    with tabs[0]:
        data_preparation_tab()
    with tabs[1]:
        training_tab()
    with tabs[2]:
        evaluation_tab()

if __name__ == "__main__":
    main()