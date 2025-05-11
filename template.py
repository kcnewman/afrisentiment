import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]:%(message)s:")

project_name = "afrisentiment"

list_of_files = [
    "./data/.gitkeep",
    "./notebooks/naive_bayes.ipynb",
    "./src/__init__.py",
    "./src/preprocessing.py",
    "./src/naive_bayes.py",
    "./src/utils.py",
    "./results/.gitkeep",
    "./requirements.txt",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, filename = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {filename}")

    if (not file_path.exists()) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating file: {file_path}")
    else:
        logging.info(f"{filename} already exists.")
