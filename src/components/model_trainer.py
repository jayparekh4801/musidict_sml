import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import lightning as L
L.seed_everything(7, workers=True)
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.components import data_loading
from src.components import data_ingestion
from src import constants
from src import utils
from src.components.array_column_transformer import ArrayColumnTransformer


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class MusicSuccessPredictor(L.LightningModule):
    def __init__(self, loss_fn=nn.CrossEntropyLoss(), learning_rate=0.001, dropout_prob=0.3):
        super(MusicSuccessPredictor, self).__init__()
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.val_loss = []
        self.train_loss = []
        
        # 2D Feature Branch (e.g., mel_spectrogram, mfccs, chroma, etc.)
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(constants.MAX_POOL_2D),
            nn.Conv2d(16, 32, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(constants.AVG_POOL_2D),
            nn.Flatten(),  # To flatten for concatenation later
        )
        
        self.mfcc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(constants.MAX_POOL_2D),
            nn.Conv2d(16, 32, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(constants.AVG_POOL_2D),
            nn.Flatten(),  # To flatten for concatenation later
        )
        
        self.chroma_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(constants.MAX_POOL_2D),
            nn.Conv2d(16, 32, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(constants.AVG_POOL_2D),
            nn.Flatten(),  # To flatten for concatenation later
        )
        
        self.spectral_contrast_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(constants.MAX_POOL_2D),
            nn.Conv2d(16, 32, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(constants.AVG_POOL_2D),
            nn.Flatten(),  # To flatten for concatenation later
        )
        
        self.tonnetz_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(constants.MAX_POOL_2D),
            nn.Conv2d(16, 32, kernel_size=constants.KERNEL_SIZE_2D, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(constants.AVG_POOL_2D),
            nn.Flatten(),  # To flatten for concatenation later
        )

        self.zcr_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=constants.KERNEL_SIZE_1D, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(constants.AVG_POOL_1D),
            nn.Flatten()
        )
        self.spectral_centroid_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=constants.KERNEL_SIZE_1D, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(constants.AVG_POOL_1D),
            nn.Flatten()
        )
        self.spectral_bandwidth_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=constants.KERNEL_SIZE_1D, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(constants.AVG_POOL_1D),
            nn.Flatten()
        )
        self.rms_energy_branch = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=constants.KERNEL_SIZE_1D, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(constants.AVG_POOL_1D),
            nn.Flatten()
        )

        # Scalar Branch
        self.scalar_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, constants.SCALAR_OUTPUT),
            nn.ReLU()
        )

        combined_size = constants.AVG_POOL_2D[0] * constants.AVG_POOL_2D[1] * 32 * 5 + constants.AVG_POOL_1D * 8 * 4 + constants.SCALAR_OUTPUT
        
        # Fully Connected Layers for Final Combined Output
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 output classes (hit, flop, can't say)
        )

    def forward(self, mel, mfcc, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features):
        # 2D Feature Processing
        mel_out = self.mel_conv(mel)
        mfcc_out = self.mfcc_conv(mfcc)
        chroma_out = self.chroma_conv(chroma)
        spectral_contrast_out = self.spectral_contrast_conv(spectral_contrast)
        tonnetz_out = self.tonnetz_conv(tonnetz)
        
        zcr_out = self.zcr_branch(zcr)
        spectral_centroid_out = self.spectral_centroid_branch(spectral_centroid)
        spectral_bandwidth_out = self.spectral_bandwidth_branch(spectral_bandwidth)
        rms_energy_out = self.rms_energy_branch(rms_energy)

        scalar_out = self.scalar_branch(scalar_features)
        combined = torch.cat((mel_out, mfcc_out, chroma_out, spectral_contrast_out, tonnetz_out, 
                              zcr_out, spectral_centroid_out, spectral_bandwidth_out, rms_energy_out, scalar_out), dim=1)
        
        # Pass through fully connected layers
        output = self.fc(combined)
        return F.softmax(output, dim=1)


    def training_step(self, batch, batch_index):
        mel_spectrogram = batch['mel_spectrogram']
        mfccs = batch['mfccs']
        chroma = batch['chroma']
        spectral_contrast = batch['spectral_contrast']
        tonnetz = batch['tonnetz']
        zcr = batch['zcr']          # Shape: (batch_size, 1, 937)
        spectral_centroid = batch['spectral_centroid'] # Shape: (batch_size, 1, 937)
        spectral_bandwidth = batch['spectral_bandwidth'] # Shape: (batch_size, 1, 937)
        rms_energy = batch['rms_energy']         # Shape: (batch_size, 1, 937)
        
        scalar_features = torch.stack((
                batch['bit_rate'].float(),   # Scalar (batch_size, 1)
                batch['duration'].float(),   # Scalar (batch_size, 1)
                batch['genre'].float()      # Categorical scalar (batch_size, 1)
            ), dim=1)
        
        labels = batch['success']
        outputs = self.forward(mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features)

        # Calculate loss
        loss = self.loss_fn(outputs, labels)

        # Backward pass and optimization
        self.train_loss.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss_mean", torch.mean(torch.tensor(self.train_loss)).item(), sync_dist=True)
        self.print("train_loss_mean", torch.mean(torch.tensor(self.train_loss)).item())
        self.train_loss = []
    
    def validation_step(self, batch, batch_index):
        mel_spectrogram = batch['mel_spectrogram']
        mfccs = batch['mfccs']
        chroma = batch['chroma']
        spectral_contrast = batch['spectral_contrast']
        tonnetz = batch['tonnetz']
        zcr = batch['zcr']          # Shape: (batch_size, 1, 937)
        spectral_centroid = batch['spectral_centroid'] # Shape: (batch_size, 1, 937)
        spectral_bandwidth = batch['spectral_bandwidth'] # Shape: (batch_size, 1, 937)
        rms_energy = batch['rms_energy']         # Shape: (batch_size, 1, 937)
        
        scalar_features = torch.stack((
                batch['bit_rate'].float(),   # Scalar (batch_size, 1)
                batch['duration'].float(),   # Scalar (batch_size, 1)
                batch['genre'].float()      # Categorical scalar (batch_size, 1)
            ), dim=1)
        
        labels = batch['success']
        outputs = self.forward(mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features)

        # Calculate loss
        loss = self.loss_fn(outputs, labels)

        # Backward pass and optimization
        self.val_loss.append(loss.item())
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss_mean", torch.mean(torch.tensor(self.val_loss)).item(), sync_dist=True)
        self.print("val_loss_mean", torch.mean(torch.tensor(self.val_loss)).item())
        self.val_loss = []
    
    def predict_step(self, batch):
        mel_spectrogram = batch['mel_spectrogram']
        mfccs = batch['mfccs']
        chroma = batch['chroma']
        spectral_contrast = batch['spectral_contrast']
        tonnetz = batch['tonnetz']
        zcr = batch['zcr']          # Shape: (batch_size, 1, 937)
        spectral_centroid = batch['spectral_centroid'] # Shape: (batch_size, 1, 937)
        spectral_bandwidth = batch['spectral_bandwidth'] # Shape: (batch_size, 1, 937)
        rms_energy = batch['rms_energy']         # Shape: (batch_size, 1, 937)
        
        scalar_features = torch.stack((
                batch['bit_rate'].float(),   # Scalar (batch_size, 1)
                batch['duration'].float(),   # Scalar (batch_size, 1)
                batch['genre'].float()      # Categorical scalar (batch_size, 1)
            ), dim=1)
        
        labels = batch['success']
        outputs = self.forward(mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features)

        # Calculate loss
        loss = self.loss_fn(outputs, labels)

        # Backward pass and optimization
        print(loss.item())
        return loss
    
    def lr_scheduler_step(self, scheduler, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        lr_scheduler = {
            'monitor': 'val_loss_mean',
            'interval': 'epoch'
        }
        return ([optimizer])

def create_confusion_matrix(model, transformation_obj_path, test_dataset, batch_size):
    transformation_obj = utils.load_object(transformation_obj_path)
    y_org = []
    y_pred = []
    output_transformer = transformation_obj.named_transformers_['cat_transforms']
    
    for batch in test_dataset:
        mel_spectrogram = batch['mel_spectrogram']
        mfccs = batch['mfccs']
        chroma = batch['chroma']
        spectral_contrast = batch['spectral_contrast']
        tonnetz = batch['tonnetz']
        zcr = batch['zcr']          # Shape: (batch_size, 1, 937)
        spectral_centroid = batch['spectral_centroid'] # Shape: (batch_size, 1, 937)
        spectral_bandwidth = batch['spectral_bandwidth'] # Shape: (batch_size, 1, 937)
        rms_energy = batch['rms_energy']         # Shape: (batch_size, 1, 937)
        
        scalar_features = torch.stack((
                batch['bit_rate'].float(),   # Scalar (batch_size, 1)
                batch['duration'].float(),   # Scalar (batch_size, 1)
                batch['genre'].float()      # Categorical scalar (batch_size, 1)
            ), dim=1)

        labels = batch['success']
        outputs = model(mel_spectrogram, mfccs, chroma, spectral_contrast, tonnetz, zcr, spectral_centroid, spectral_bandwidth, rms_energy, scalar_features)

        y_temp_actual = output_transformer.inverse_transform(labels)
        y_temp_actual = np.squeeze(y_temp_actual).tolist()
        y_org.append(y_temp_actual)

        max_indices = torch.argmax(outputs, dim=1)
        y_temp_pred = torch.zeros_like(outputs)
        y_temp_pred[torch.arange(outputs.size(0)), max_indices] = 1

        y_temp_pred = output_transformer.inverse_transform(y_temp_pred)
        # print(y_temp_pred)
        y_temp_pred = np.squeeze(y_temp_pred).tolist()
        # print(y_temp_pred)
        y_pred.append(y_temp_pred)

    y_org = np.concatenate(y_org, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    categories = sorted(set(y_org) | set(y_pred))

    # Generate the confusion matrix
    matrix = confusion_matrix(y_org, y_pred, labels=categories)

    # Convert to a DataFrame for better readability
    confusion_df = pd.DataFrame(matrix, index=categories, columns=categories)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=categories, yticklabels=categories)

    # Add labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap')

    # Save the figure as an image (e.g., PNG format)
    plt.savefig(os.path.join(os.getcwd(), 'artifacts/plots/confusion_matrix.png'), dpi=300)
    
    
if __name__ == "__main__":
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    dropout_prob = 0.3
    lr_logger = LearningRateMonitor()
    model_checkpoint = ModelCheckpoint(dirpath="./artifacts/MODELS",save_last=True, save_top_k=3, monitor="val_loss_mean")
    epochs = 60
    data_loader_obj = data_loading.DataModule(batch_size=batch_size)
    train_loader = data_loader_obj.train_dataloader()
    val_loader = data_loader_obj.val_dataloader()
    test_loader = data_loader_obj.val_dataloader()

    lightning_model = MusicSuccessPredictor(loss_fn=criterion, learning_rate=learning_rate, dropout_prob=dropout_prob)

    trainer = L.Trainer(max_epochs=epochs, callbacks=[lr_logger, model_checkpoint])
    trainer.fit(lightning_model, train_loader, val_loader)

    create_confusion_matrix(lightning_model, os.path.join(os.getcwd(), "artifacts", "transformation.pkl"), test_loader, batch_size)


    
    