import time

class ModelTrainingWorker:
    def __init__(self, training_data_file, label_mapping_mode, synonyms_list, embedding_model, model_type, hyperparams, progress_callback, status_callback):
        self.training_data_file = training_data_file
        self.label_mapping_mode = label_mapping_mode
        self.synonyms_list = synonyms_list
        self.embedding_model = embedding_model
        self.model_type = model_type
        self.hyperparams = hyperparams
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def run(self):
        try:
            self.status_callback("Starting model training...")

            # Simulate progress
            for i in range(101):
                time.sleep(0.01)
                self.progress_callback(i)

            self.status_callback("Model training complete!")
            return True
        except Exception as e:
            self.status_callback(f"Error during model training: {e}")
            return False