import numpy as np


LISTENS_SPLIT = np.array([0, 292, 2018, np.inf])

NUMBER_TO_GENRE_MAPPING = {
    38: "Experimental",
    15: "Electronic",
    12: "Rock",
    1235: "Instrumental",
    10: "Pop",
    17: "Folk",
    21: "Hip-Hop",
    2: "International",
    4: "Jazz",
    5: "Classical",
    9: "Country",
    20: "Spoken",
    3: "Blues",
    14: "Soul-RnB",
    8: "Historic",
    13: "Easy Listening"
}

NUMERICAL_DATA = ["mel_spectrogram", 
                  "mfccs",
                  "chroma",
                  "spectral_contrast",
                  "zcr",
                  "spectral_centroid",
                  "spectral_bandwidth",
                  "rms_energy",
                  "tonnetz"]
CATEGORICAL_DATA = ["genres", "listens"]

TIME_SERIES_DATA_SHAPES = {
    "original_audio":(479626,),
    "mel_spectrogram":(128, 937),
    "mfccs":(13, 937),
    "chroma":(12, 937),
    "spectral_contrast":(7, 937),
    "zcr":(1, 937),
    "spectral_centroid":(1, 937),
    "spectral_bandwidth":(1, 937),
    "rms_energy":(1, 937),
    "tonnetz":(6, 937),
}

# TIME_SERIES_DATA_SHAPES = {
#     "original_audio":(479626,),
#     "mel_spectrogram":(128, 1024),
#     "mfccs":(8, 1024),
#     "chroma":(8, 1024),
#     "spectral_contrast":(8, 1024),
#     "zcr":(1, 1024),
#     "spectral_centroid":(1, 1024),
#     "spectral_bandwidth":(1, 1024),
#     "rms_energy":(1, 1024),
#     "tonnetz":(8, 1024),
# }

FEATURE_COLUMNS = ["genre",
                "bit_rate",
                "duration",
                "mel_spectrogram", 
                "mfccs",
                "chroma",
                "spectral_contrast",
                "zcr",
                "spectral_centroid",
                "spectral_bandwidth",
                "rms_energy",
                "tonnetz",]

AVG_POOL_2D = (4, 256)
AVG_POOL_1D = 256
SCALAR_OUTPUT = 128
KERNEL_SIZE_2D = (2, 2)
KERNEL_SIZE_1D = 2
MAX_POOL_2D = (1, 1)