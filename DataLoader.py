import pandas as pd

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_dataset(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def get_data(self):
        return self.data

