import librosa
import numpy as np
import os
from scipy.stats import kurtosis, skew
import pandas as pd
from audioread import NoBackendError
from soundfile import LibsndfileError

bird_species = os.listdir('dataset/6_classes_audio')

num_species = len(bird_species)     # liczba gatunków
num_samples = 200   # ilosc probek na kazda klase


col = ['Name', *range(1, 161)]
df = pd.DataFrame(columns=col)

# wypelnianie pustej ramki danych zadana iloscia probek
for spec_name in bird_species:
    files = os.listdir('dataset/6_classes_audio/' + spec_name)
    i = num_samples
    for file in files:
        if i > 0:
            try:
                y, sr = librosa.load('dataset/6_classes_audio/' + spec_name + '/' + file)
                yt, index = librosa.effects.trim(y)     # wycinanie ciszy z konca i poczatku pliku

                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                mfcc_features = np.array([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.median(mfcc, axis=1),
                                          np.max(mfcc, axis=1), np.min(mfcc, axis=1), kurtosis(mfcc, axis=1),
                                          skew(mfcc, axis=1), np.var(mfcc, axis=1)])

                mfcc_features = mfcc_features.flatten()
                mfcc_features = mfcc_features.tolist()

                mfcc_features.insert(0, spec_name)
                df.loc[len(df)] = mfcc_features

                i -= 1
            except (LibsndfileError, EOFError, NoBackendError) as e:
                print(e)

df.to_csv('dataset/features/extracted_features_6_classes.csv', index=False)
