import os
import pickle

import pandas as pd
from keras_tuner import BayesianOptimization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from bin.utils.build_model import build_model
from bin.utils.load_data_from_arrays import load_data_from_arrays

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path_for_models = 'models_ai/01122021/'
num_words = 5000
max_words = 50
baza = pd.read_csv('../data/avoska_ai.csv', header=None, names=['category', 'text'])
baza = baza.sample(frac=1).reset_index(drop=True)

all_text = baza['text']
all_cat = baza['category']

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(all_text)
total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))
# Сохраняем токенайзер
with open(path_for_models + 'tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Загружаем токенайзер
# with open('tokenizer.pickle', 'rb') as f:
#    loaded_tokenizer = pickle.load(f)

# Делим данные на две части
x_train, y_train, x_test, y_test = load_data_from_arrays(all_text, all_cat, train_test_split=0.8)

# Секвенируем текст в цифры
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
# Обрезаем длину строк по максимальному количеству слов
x_train = pad_sequences(x_train, maxlen=max_words)
x_test = pad_sequences(x_test, maxlen=max_words)


tuner = BayesianOptimization(
    build_model,  # функция создания модели
    objective='val_accuracy',  # метрика, которую нужно оптимизировать -
    # доля правильных ответов на проверочном наборе данных
    max_trials=200,  # максимальное количество запусков обучения
    directory='test_directory'  # каталог, куда сохраняются обученные сети
)

# tuner.search_space_summary()

tuner.search(x_train,  # Данные для обучения
             y_train,  # Правильные ответы
             batch_size=64,  # Размер мини-выборки
             epochs=40,  # Количество эпох обучения
             validation_data=(x_test, y_test)
             )

tuner.results_summary()

models = tuner.get_best_models(num_models=3)
name_best_models = ("best_model1.h5", "best_model2.h5", "best_model3.h5")
for idx, model in enumerate(models):
    model.summary()
    model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    model.save(path_for_models + name_best_models[idx])
    print()
