import time

class ChallengeWorker:
    def __init__(self, trained_model_path, secret_test_set_path, progress_callback, status_callback):
        self.trained_model_path = trained_model_path
        self.secret_test_set_path = secret_test_set_path
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def run(self):
        try:
            self.status_callback("Starting Challenge predictions...")

            # Simulate progress
            for i in range(101):
                time.sleep(0.01)
                self.progress_callback(i)

            self.status_callback("Challenge predictions complete!")
            return True
        except Exception as e:
            self.status_callback(f"Error during challenge: {e}")
            return False