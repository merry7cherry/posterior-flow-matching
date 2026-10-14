import os
import pickle

def load_pkl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_base_data_dir() -> str:
    # get the environment variable DATA_DIR
    data_dir = os.environ.get("DATA_DIR", None)

    if data_dir is None:
        raise ValueError("DATA_DIR environment variable is not set.")

    return data_dir
