import os
import numpy as np
from dataclasses import dataclass
import torch

from torch.utils.data import DataLoader, Dataset
import lightning as L
from sklearn.model_selection import train_test_split

from src import utils

@dataclass
class DataLoaderConfig:
    # transformed_data_path = os.path.join(os.getcwd(), "data/transformed_data", "transformed_dataset.npy")
    transformed_data_path = os.path.join("/home/jparekh4/musidict", "data/transformed_data", "transformed_dataset.npy")
    


class MusicDataset(Dataset):
    def __init__(self, features):
        self.features = features  # Dictionary with all feature arrays
        

    def __getitem__(self, idx):
        mel_spectrogram = self.features['mel_spectrogram'][idx].clone().detach().unsqueeze(0).float()
        mfccs = self.features['mfccs'][idx].clone().detach().unsqueeze(0).float()
        chroma = self.features['chroma'][idx].clone().detach().unsqueeze(0).float()
        spectral_contrast = self.features['spectral_contrast'][idx].clone().detach().unsqueeze(0).float()
        zcr = self.features['zcr'][idx].clone().detach().float()
        spectral_centroid = self.features['spectral_centroid'][idx].clone().detach().float()
        spectral_bandwidth = self.features['spectral_bandwidth'][idx].clone().detach().float()
        rms_energy = self.features['rms_energy'][idx].clone().detach().float()
        tonnetz = self.features['tonnetz'][idx].clone().detach().unsqueeze(0).float()
        # Scalar features
        bit_rate = self.features["bit_rate"][idx].clone().detach().float()
        duration = self.features["duration"][idx].clone().detach().float()
        genre = self.features["genre"][idx].clone().detach().float()

        target = self.features["success"][idx].clone().detach().float() # One-hot encoded target

        return {
            'mel_spectrogram': mel_spectrogram,
            'mfccs': mfccs,
            'chroma': chroma,
            'spectral_contrast': spectral_contrast,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'rms_energy': rms_energy,
            'tonnetz': tonnetz,
            'bit_rate': bit_rate,
            'duration': duration,
            'genre': genre,
            'success': target
        }

    def __len__(self):
        return len(self.features['mel_spectrogram'])


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.transformed_data = DataLoaderConfig()
        self.load_and_split_data()
    
    def load_and_split_data(self):
        self.dataset = np.load(self.transformed_data.transformed_data_path, allow_pickle=True)
        # self.dataset = np.load("/Users/jayparekh/Documents/projects/musidict/data/transformed_data/transformed_dataset.npy", allow_pickle=True)
        train_data, temp_data = train_test_split(self.dataset, test_size=0.2, random_state=42, shuffle=True)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        self.train_data = utils.convert_dataset_into_tensor_dict(train_data)
        self.val_data = utils.convert_dataset_into_tensor_dict(val_data)
        self.test_data = utils.convert_dataset_into_tensor_dict(test_data)
    
    def train_dataloader(self):
        train_data = MusicDataset(self.train_data)
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=3)
    
    def val_dataloader(self):
        val_data = MusicDataset(self.val_data)
        return DataLoader(val_data, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        test_data = MusicDataset(self.test_data)
        return DataLoader(test_data, batch_size=self.batch_size, num_workers=3)