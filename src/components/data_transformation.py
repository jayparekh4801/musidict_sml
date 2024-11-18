import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


import torch

from src.exception import CustomException
from src.logger import logging
from src.components import data_ingestion
from src.components.array_column_transformer import ArrayColumnTransformer
from src import constants
from src import utils

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    tranformationObjectPath:  str = "artifacts/transformation.pkl"
    transformedDataPathDir: str = "data/transformed_data/"


class DataTransformation:

    def __init__(self):
        self.transformation_artifacts = DataTransformationConfig()
    
    def get_transformation_object(self):
        try:
            cat_pipeline = Pipeline([
                ("cat_transform", OneHotEncoder())
            ])

            num_pipeline = Pipeline([
                ("num_pipeline", StandardScaler())
            ])

            time_series_pipeline = Pipeline([
                ("time_series_pipeline", ArrayColumnTransformer())
            ])

            # Define preprocessor
            preprocessor = ColumnTransformer(transformers=[
                ("num_transforms", num_pipeline, ["bit_rate", "duration"]),
                ("time_series_transforms", time_series_pipeline, ['mel_spectrogram', 'mfccs',
                'chroma', 'spectral_contrast', 'zcr', 'spectral_centroid',
                'spectral_bandwidth', 'rms_energy', 'tonnetz']),
                ("cat_transforms", cat_pipeline, ["success"]),
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self):
        try:
            logging.info("Data Transformation Started.")
            dataset_df = data_ingestion.DataIngestion().initiate_data_ingestion()
            preprocessing_obj=self.get_transformation_object()
            transformed_dataset = preprocessing_obj.fit_transform(dataset_df)
            transformed_dataset = np.column_stack((dataset_df["genre"].values, transformed_dataset))
            logging.info("Data Transformation Finised.")
            logging.info("Saving transformatino object.")
            utils.save_object(
                file_path=os.path.join(os.getcwd(), self.transformation_artifacts.tranformationObjectPath),
                obj=preprocessing_obj
            )

            logging.info("Saving transformed data.")
            logging.info(f"Transformed Data Shape: {transformed_dataset.shape}")

            transformed_data_file = os.path.join(os.getcwd(), self.transformation_artifacts.transformedDataPathDir, "transformed_dataset.npy")
            logging.info(f"output saved to {transformed_data_file}")
            np.save(transformed_data_file, transformed_dataset)

            return (transformed_data_file, transformed_dataset)

            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_transformation_obj = DataTransformation()
    data_transformation_obj.initiate_data_transformation()

        









