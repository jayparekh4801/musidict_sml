import numpy as np
import pandas as pd
import os
import pickle
import sys
import torch
import librosa
from src.exception import CustomException
from src.logger import logging
from src import constants
from src.components.array_column_transformer import ArrayColumnTransformer

def categorize_listens(val):
    if constants.LISTENS_SPLIT[0] <= val < constants.LISTENS_SPLIT[1]:
        return "flop"
    elif constants.LISTENS_SPLIT[1] <= val < constants.LISTENS_SPLIT[2]:
        return "average"
    elif constants.LISTENS_SPLIT[2] <= val < constants.LISTENS_SPLIT[3]:  # Last boundary is np.inf, no upper limit
        return "hit"


def preprocess_genres(val):
    for i in val:
        if i in constants.NUMBER_TO_GENRE_MAPPING.keys():
            return i
    
    return None

def normalize_array_shape(array, target_shape):
    # Truncate if the array is larger than the target shape
    truncated_array = array[:target_shape[0], :target_shape[1]]
    
    # Pad if the array is smaller than the target shape
    padding = ((0, max(0, target_shape[0] - truncated_array.shape[0])),
               (0, max(0, target_shape[1] - truncated_array.shape[1])))
    
    normalized_array = np.pad(truncated_array, padding, mode='constant')
    return normalized_array


def reshape_all_time_series_data(data):
    reshaped_data = {}
    for key, val in data.items():
        if key == "original_audio":
            continue
        if key == "metadata":
            break
        if key in constants.TIME_SERIES_DATA_SHAPES.keys():
            reshaped_data[key] = normalize_array_shape(val, constants.TIME_SERIES_DATA_SHAPES[key])
    
    return reshaped_data


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        print(dir_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def convert_dataset_into_tensor_dict(data):
    success = torch.tensor(data[:, -3:].astype(float))
    result_dict = {}
    for ind, col in enumerate(constants.FEATURE_COLUMNS):
        result_dict[col] = torch.tensor(np.stack(data[:, ind]), dtype=torch.float)

    
    result_dict["success"] = success

    return result_dict


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def preprocess_file(data):
        data_point = []
        if "metadata" not in data.keys():
            return None
            
        genre = preprocess_genres(data["metadata"][0]["genres"])
        if not genre:
            return None

        data_point.append(genre)
        data_point.append(data["metadata"][0]["bit_rate"])
        data_point.append(data["metadata"][0]["duration"])
        data_point.append(categorize_listens(data["metadata"][0]["listens"]))
        

        data = reshape_all_time_series_data(data)
        data_point.append(np.array(data["mel_spectrogram"]))
        data_point.append(np.array(data["mfccs"]))
        data_point.append(np.array(data["chroma"]))
        data_point.append(np.array(data["spectral_contrast"]))
        data_point.append(np.array(data["zcr"]))
        data_point.append(np.array(data["spectral_centroid"]))
        data_point.append(np.array(data["spectral_bandwidth"]))
        data_point.append(np.array(data["rms_energy"]))
        data_point.append(np.array(data["tonnetz"]))

        return data_point


def process_music(music_data):
    print(music_data["file_path"])
    y, sr = librosa.load(os.path.join(os.getcwd(), music_data["file_path"]), sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=200.0)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rms_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for key, val in constants.NUMBER_TO_GENRE_MAPPING.items():
        if val == music_data["genre"]:
            genre = key
            break

    data = {
                "mel_spectrogram": mel_spectrogram,
                "mfccs": mfccs,
                "chroma": chroma,
                "spectral_contrast": spectral_contrast,
                "zcr": zcr,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "rms_energy": rms_energy,
                "tonnetz": tonnetz,
                "metadata":[{
                    "genres": [genre],
                    "bit_rate": music_data["bit_rate"],
                    "duration": music_data["duration"],
                    "listens": 0,
                }]
            }

    preprocessed_data = preprocess_file(data)
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

    return data_df
