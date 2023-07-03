import librosa
import numpy as np
import os
from scipy.stats import kurtosis, skew
import pandas as pd
from audioread import NoBackendError
from soundfile import LibsndfileError

bird_species = os.listdir('dataset/6_classes_audio')

num_species = len(bird_species)     # liczba gatunków

for spec_name in bird_species:
    files = os.listdir('dataset/6_classes_audio/' + spec_name)
    list_samples = []
    for file in files:
        try:
            y, sr = librosa.load('dataset/6_classes_audio/' + spec_name + '/' + file)
            dur = librosa.get_duration(y=y, sr=sr)

            if dur >= 10:
                list_samples.append(file)

        except (LibsndfileError, EOFError, NoBackendError) as e:
            print(e)
    print(spec_name)
    print(len(list_samples))
    print('---')
