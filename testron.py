import os
import pickle

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


data = pd.read_csv('data/avoska_test_morfy.csv', header=None, names=['category', 'text'])

with open('BestModels/85_20_93/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Откладываем для теста ВСЕ данные сырые оттокенизированные
x_data_all = data['text'].tolist()
y_data_all = np.asarray(data['category'].tolist())
x_data_all = tokenizer.texts_to_sequences(x_data_all)
x_data_all = pad_sequences(x_data_all, maxlen=100)

model = keras.models.load_model("BestModels/85_20_93/best_model0.h5")
model.summary()
model.evaluate(x_data_all, y_data_all)

