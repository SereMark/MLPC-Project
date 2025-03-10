class DataPreparationWorker:
    def __init__(self, raw_audio_dir, annotations_file, processed_data_dir, sample_rate, progress_callback, status_callback):
        self.raw_audio_dir=raw_audio_dir,
        self.annotations_file=annotations_file,
        self.processed_data_dir=processed_data_dir,
        self.sample_rate=sample_rate,
        self.progress_callback=progress_callback,
        self.status_callback=status_callback

    def run(self):
        raise NotImplementedError("DataPreparationWorker.run() not yet implemented.")