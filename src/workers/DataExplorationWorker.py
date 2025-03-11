import time

class DataExplorationWorker:
    def __init__(self, features_file, clustering_method, n_clusters, embedding_type, progress_callback, status_callback):
        self.features_file = features_file
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.embedding_type = embedding_type
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def run(self):
        try:
            self.status_callback("Starting data exploration...")

            # Simulate progress
            for i in range(101):
                time.sleep(0.01)
                self.progress_callback(i)

            self.status_callback("Data exploration complete!")
            return True
        except Exception as e:
            self.status_callback(f"Error during data exploration: {e}")
            return False