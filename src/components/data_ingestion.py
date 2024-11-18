import os
import sys
import numpy as np
import pandas as pd
import torch
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from zipfile import BadZipFile

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src import utils
from src import constants

@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/raw_data"


class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_dict = {}

    def initiate_data_ingestion(self):
            dataset = []
            logging.info("Data Ingestion Started.")
            count = 0
            for filename in tqdm(os.listdir(os.path.join(os.getcwd(), self.data_ingestion_config.raw_data_path))):
                
                if filename.endswith('.npz') and os.path.getsize(os.path.join(os.getcwd(), self.data_ingestion_config.raw_data_path, filename)):
                    try:
                        file_path = os.path.join(os.getcwd(), self.data_ingestion_config.raw_data_path, filename)
                        data = np.load(file_path, allow_pickle=True)
                        data = utils.preprocess_file(data)
                        if not data:
                            print(f"{filename} skipped.")
                            continue
                        dataset.append(data)
                        count += 1
                        print(f"Total Files Processed: {count}")
                    except (EOFError, OSError, ValueError, BadZipFile) as e:
                        print(f"{filename} skipped.")
              
            dataset = np.array(dataset, dtype=object)
            dataset_df = pd.DataFrame(dataset, columns=[
                "genre",
                "bit_rate",
                "duration",
                "success",
                "mel_spectrogram",
                "mfccs",
                "chroma",
                "spectral_contrast",
                "zcr",
                "spectral_centroid",
                "spectral_bandwidth",
                "rms_energy",
                "tonnetz",
                ])
            logging.info("Data Ingestion Finished.")
            return dataset_df


if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    _ = data_ingestion_obj.initiate_data_ingestion()
