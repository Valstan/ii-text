import pickle

import pandas as pd
from keras_tuner import BayesianOptimization
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from bin.utils.build_model import build_model
from load_data_from_arrays import load_data_from_arrays

num_words = 10000
max_words = 50
baza = pd.read_csv('data/avoska_ai.csv', header=None, names=['category', 'text'])
baza = baza.sample(frac=1).reset_index(drop=True)

all_text = baza['text']
all_cat = baza['category']

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(all_text)
total_words = len(tokenizer.word_index)
print('В словаре {} слов'.format(total_words))
# Сохраняем токенайзер
with open('models_ai/21ноября2021/tokenizer.pickle', 'wb') as handle:
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

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
# encoder = LabelEncoder()
# encoder.fit(y_train)
# y_train = encoder.transform(y_train)
# y_test = encoder.transform(y_test)
# num_classes = np.max(y_train) + 1
# print('Количество категорий для классификации: {}'.format(num_classes))

tuner = BayesianOptimization(
    build_model,  # функция создания модели
    objective='val_accuracy',  # метрика, которую нужно оптимизировать -
    # доля правильных ответов на проверочном наборе данных
    max_trials=400,  # максимальное количество запусков обучения
    directory='test_directory'  # каталог, куда сохраняются обученные сети
)

# tuner.search_space_summary()

tuner.search(x_train,  # Данные для обучения
             y_train,  # Правильные ответы
             batch_size=256,  # Размер мини-выборки
             epochs=20,  # Количество эпох обучения
             validation_split=0.1,  # Часть данных, которая будет использоваться для проверки
             )

tuner.results_summary()

models = tuner.get_best_models(num_models=3)
model_save_path = ("best_model1.h5", "best_model2.h5", "best_model3.h5")
for idx, model in enumerate(models):
    model.summary()
    model.evaluate(x_train, y_train)
    model.save(model_save_path[idx])
    print()

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
# model = load_model('my_model.h5')
