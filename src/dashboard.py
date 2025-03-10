import os, streamlit as st, DataPreparationWorker

st.set_page_config(page_title="Audio Dashboard", page_icon="🔊", layout="wide")

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
        else:
            status_text.text("⚠️ Failed.")
    except Exception as e:
        status_text.text(f"⚠️ Error: {e}")

def data_preparation_tab():
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

    st.write("")
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

if __name__ == "__main__":
    st.title("🔊 Audio Dashboard")

    tabs = st.tabs(["Data Preparation"])

    with tabs[0]:
        data_preparation_tab()