import os
import streamlit as st
import DataPreparationWorker
import TrainingWorker
import EvaluationWorker

# ---- PAGE CONFIGURATION ----
st.set_page_config(
    page_title="Audio ML Management Dashboard",
    page_icon="🔊",
    layout="wide"
)

# ---- PATH VALIDATION UTILITY ----
def validate_path(path: str, path_type: str = "file") -> bool:
    """
    Checks if a path exists and matches the specified type: 'file' or 'directory'.
    """
    if not path:
        return False
    if path_type == "file":
        return os.path.isfile(path)
    if path_type == "directory":
        return os.path.isdir(path)
    return False

# ---- INPUT FIELD WITH VALIDATION ----
def input_with_validation(
    label: str,
    default_value: str = "",
    path_type: str = "file",
    help_text: str = None
) -> str:
    """
    Creates a text input for a file or directory path and visually indicates validity.
    """
    path = st.text_input(label, default_value, help=help_text)
    if path:
        if validate_path(path, path_type):
            st.markdown("✅ **Valid Path**")
        else:
            st.markdown("⚠️ **Invalid Path**")
    return path

# ---- WORKER EXECUTION AND PROGRESS ----
def execute_worker(create_worker):
    """
    Executes a worker with live progress and status updates in the UI.
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

# ---- DATA PREPARATION TAB ----
def data_preparation_tab():
    st.subheader("Data Preparation")
    
    raw_audio_dir = input_with_validation(
        label="Raw Audio Directory:",
        default_value="data/raw_audio",
        path_type="directory",
        help_text="Path containing raw audio files."
    )
    annotations_file = input_with_validation(
        label="Annotations File:",
        default_value="data/annotations.csv",
        path_type="file",
        help_text="CSV file with labels or metadata."
    )
    processed_data_dir = input_with_validation(
        label="Processed Data Output Directory:",
        default_value="data/processed_audio",
        path_type="directory",
        help_text="Folder where processed audio files will be saved."
    )
    sample_rate = st.number_input(
        label="Sample Rate (Hz):",
        min_value=8000,
        max_value=48000,
        value=16000,
        step=1000,
        help="Sampling rate for audio processing."
    )

    st.write("")  # Spacing
    if st.button("Start Data Preparation"):
        if not validate_path(raw_audio_dir, "directory"):
            st.error("Invalid raw audio directory.")
            return
        if not validate_path(annotations_file, "file"):
            st.error("Invalid annotations file.")
            return
        if not validate_path(processed_data_dir, "directory"):
            st.error("Invalid processed data output directory.")
            return

        execute_worker(lambda progress_cb, status_cb: DataPreparationWorker(
            raw_audio_dir=raw_audio_dir,
            annotations_file=annotations_file,
            processed_data_dir=processed_data_dir,
            sample_rate=sample_rate,
            progress_callback=progress_cb,
            status_callback=status_cb
        ))

# ---- TRAINING TAB ----
def training_tab():
    st.subheader("Training")
    
    epochs = st.number_input(
        label="Epochs:",
        min_value=1,
        max_value=1000,
        value=10,
        help="Number of epochs to train."
    )
    batch_size = st.number_input(
        label="Batch Size:",
        min_value=1,
        max_value=2048,
        value=32,
        help="Number of samples per gradient update."
    )
    learning_rate = st.number_input(
        label="Learning Rate:",
        min_value=1e-6,
        max_value=1.0,
        value=0.001,
        format="%.6f",
        help="Learning rate for your optimizer."
    )

    st.markdown("---")
    train_data_path = input_with_validation(
        label="Training Data File:",
        default_value="data/processed_audio/train_data.csv",
        path_type="file",
        help_text="CSV file with training data."
    )
    val_data_path = input_with_validation(
        label="Validation Data File (optional):",
        default_value="data/processed_audio/val_data.csv",
        path_type="file",
        help_text="CSV file with validation data."
    )
    model_output_dir = input_with_validation(
        label="Model Output Directory:",
        default_value="models/",
        path_type="directory",
        help_text="Folder to save trained models."
    )

    st.write("")  # Spacing
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

# ---- EVALUATION TAB ----
def evaluation_tab():
    st.subheader("Evaluation")
    
    model_path = input_with_validation(
        label="Trained Model Path:",
        default_value="models/audio_model.pth",
        path_type="file",
        help_text="Path to the trained model file."
    )
    test_data_path = input_with_validation(
        label="Test Data File:",
        default_value="data/processed_audio/test_data.csv",
        path_type="file",
        help_text="CSV file with test data."
    )
    results_output_dir = input_with_validation(
        label="Evaluation Results Directory:",
        default_value="evaluation/",
        path_type="directory",
        help_text="Folder to store evaluation results."
    )

    st.write("")  # Spacing
    if st.button("Start Evaluation"):
        if not validate_path(model_path, "file"):
            st.error("Invalid model path.")
            return
        if not validate_path(test_data_path, "file"):
            st.error("Invalid test data file.")
            return
        if not validate_path(results_output_dir, "directory"):
            st.error("Invalid output directory for results.")
            return

        execute_worker(lambda progress_cb, status_cb: EvaluationWorker(
            model_path=model_path,
            test_data_path=test_data_path,
            results_output_dir=results_output_dir,
            progress_callback=progress_cb,
            status_callback=status_cb
        ))

# ---- MAIN APP ----
def main():
    st.title("🔊 Audio ML Management Dashboard")

    tabs = st.tabs(["Data Preparation", "Training", "Evaluation"])

    with tabs[0]:
        data_preparation_tab()
    with tabs[1]:
        training_tab()
    with tabs[2]:
        evaluation_tab()

if __name__ == "__main__":
    main()