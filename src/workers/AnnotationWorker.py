import time

class AnnotationWorker:
    def __init__(self, label_studio_project_path, annotation_config_file, progress_callback, status_callback):
        self.label_studio_project_path = label_studio_project_path
        self.annotation_config_file = annotation_config_file
        self.progress_callback = progress_callback
        self.status_callback = status_callback

    def run(self):
        try:
            self.status_callback("Starting annotation process...")

            # Simulate progress
            for i in range(101):
                time.sleep(0.01)
                self.progress_callback(i)

            self.status_callback("Annotation process complete!")
            return True
        except Exception as e:
            self.status_callback(f"Error during annotation: {e}")
            return False