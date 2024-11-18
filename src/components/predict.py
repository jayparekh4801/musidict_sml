import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.components import data_ingestion
from src.components.data_transformation import ArrayColumnTransformer
from src.components import model_trainer
from src import utils


class PredictModuleConfig:
    artifacts = os.path.join(os.getcwd(), "artifacts")
    transformation_object_file = "transformation.pkl"
    predict_file = "data/raw_data/npz_files_11111.npz"
    model_file = "MODELS/epoch=56-step=5871.ckpt"

class PredictModule:

    def __init__(self):
        self.predict_config = PredictModuleConfig()
        self.transformer_obj_file = os.path.join(self.predict_config.artifacts, self.predict_config.transformation_object_file)
        self.transfrmation_obj = utils.load_object(self.transformer_obj_file)

    def performTransformation(self):
        data = np.load(os.path.join(os.getcwd(), self.predict_config.predict_file), allow_pickle=True)
        preprocessed_data = utils.preprocess_file(data)
        print(preprocessed_data)
        preprocessed_data = np.array(preprocessed_data, dtype=object).reshape((1, 13))
        data_df = pd.DataFrame(preprocessed_data, columns=[
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

        transformed_data = self.transfrmation_obj.transform(data_df)
        transformed_data = np.column_stack((data_df["genre"].values, transformed_data))
        return utils.convert_dataset_into_tensor_dict(transformed_data)
    
    def prepare_model(self):
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.001
        dropout_prob = 0.3
        checkpoint = torch.load(os.path.join(self.predict_config.artifacts, self.predict_config.model_file), 
                                map_location=lambda storage, loc: storage)
        model = model_trainer.MusicSuccessPredictor(loss_fn=criterion,learning_rate=learning_rate,dropout_prob=dropout_prob)
        model.load_state_dict(state_dict=checkpoint["state_dict"])

        # trainer = L.trainer()
        # trainer.test(os.path.join())
        return model

    def predict(self):
        transformed_data = self.performTransformation()
        model = self.prepare_model()
        mel_spectrogram = transformed_data['mel_spectrogram'].unsqueeze(0)
        mfccs = transformed_data['mfccs'].unsqueeze(0)
        chroma = transformed_data['chroma'].unsqueeze(0)
        spectral_contrast = transformed_data['spectral_contrast'].unsqueeze(0)
        tonnetz = transformed_data['tonnetz'].unsqueeze(0)
        zcr = transformed_data['zcr']          # Shape: (batch_size, 1, 937)
        spectral_centroid = transformed_data['spectral_centroid'] # Shape: (batch_size, 1, 937)
        spectral_bandwidth = transformed_data['spectral_bandwidth'] # Shape: (batch_size, 1, 937)
        rms_energy = transformed_data['rms_energy']         # Shape: (batch_size, 1, 937)
        
        scalar_features = torch.stack((
                transformed_data['bit_rate'].float(),   # Scalar (batch_size, 1)
                transformed_data['duration'].float(),   # Scalar (batch_size, 1)
                transformed_data['genre'].float()      # Categorical scalar (batch_size, 1)
            ), dim=1)
        print(transformed_data["success"])
        output = model(mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features)

        one_hot_output = torch.zeros(size=(1, 3))

        one_hot_output[:,torch.argmax(output)] = 1

        output_transformer = self.transfrmation_obj.named_transformers_['cat_transforms']

# Perform the inverse transformation using the output transformer
        print(output_transformer.inverse_transform(one_hot_output))
        
        return output
        
        
if __name__ == "__main__":
    # model = PredictModule().prepare_model()
    # data_loader_obj = data_loading.DataModule(batch_size=1)
    # test_loader = data_loader_obj.test_dataloader()
    # output = model(test_loader)
    model = PredictModule()
    print(model.predict())
