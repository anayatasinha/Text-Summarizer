import os 
from datasets import load_dataset
import urllib.request as request
import zipfile
from textSummarizer.logging import logger
import importlib
import textSummarizer.utils.common as common
importlib.reload(common)
from textSummarizer.utils.common import get_size
from pathlib import Path
from textSummarizer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self , config: DataIngestionConfig):
        self.config = config

    def download_file(self):

        # Case 1: Hugging Face dataset name instead of URL
        if self.config.source_URL.startswith("hf://"):
            dataset_name = self.config.source_URL.replace("hf://", "")
            ds = load_dataset(dataset_name)

            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)

            # Save dataset splits into a zip file
            with zipfile.ZipFile(self.config.local_data_file, 'w') as zipf:
                for split in ds.keys():
                    filename = f"{split}.json"
                    path = os.path.join(os.path.dirname(self.config.local_data_file), filename)
                    ds[split].to_json(path)
                    zipf.write(path, arcname=filename)

        # Case 2: Normal URL download
        else:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL,
                    filename=self.config.local_data_file
                )
                logger.info(f"{filename} downloaded with info: \n{headers}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path:str
        Extracts the zip file into the data directory
        Funtion returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)