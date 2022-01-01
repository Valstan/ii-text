import os
import pickle

import gensim
import keras_tuner as kt
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import normalize

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
if optimizer == 'adam':
    optimizer = Adam(learning_rate=hp_learning_rate)
elif optimizer == 'rmsprop':
    optimizer = RMSprop(learning_rate=hp_learning_rate)
elif optimizer == 'nadam':
    optimizer = Nadam(learning_rate=hp_learning_rate)
else:
    raise
'''


def model_builder(hp):
    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'elu', 'selu'])

    model = Sequential()
    model.add(Embedding(input_dim=dict_len + 1,
                        output_dim=64,
                        mask_zero=True))
    model.add(Dropout(rate=hp.Float('dropout1',
                                    max_value=0.8,
                                    min_value=0.5,
                                    step=0.1)
                      ))
    model.add(GRU(units=hp.Int('gru1units',
                               max_value=8,
                               min_value=2,
                               step=2),
                  activation=activation_choice,
                  dropout=hp.Float('gru1drop',
                                   max_value=0.8,
                                   min_value=0.5,
                                   step=0.1)
                  ))
    model.add(Dropout(rate=hp.Float('dropout1',
                                    max_value=0.8,
                                    min_value=0.5,
                                    step=0.1)
                      ))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam'])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=hp_learning_rate)
    elif optimizer == 'nadam':
        optimizer = Nadam(learning_rate=hp_learning_rate)
    else:
        raise
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics='binary_accuracy')
    return model


# def model_builder(hp):
#     activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'elu', 'selu'])
#
#     model = Sequential()
#     model.add(Embedding(input_dim=dict_len,
#                         output_dim=32,
#                         input_length=string_len,
#                         mask_zero=True))
#     model.add(Flatten())
#     model.add(Dropout(rate=hp.Float('dropout0',
#                                     max_value=0.5,
#                                     min_value=0.2,
#                                     step=0.1)))
#     model.add(Dense(units=hp.Int('dense1units',
#                                  max_value=16,
#                                  min_value=2,
#                                  step=2),
#                     activation=activation_choice
#                     ))
#     model.add(Dropout(rate=hp.Float('dropout1',
#                                     max_value=0.3,
#                                     min_value=0.1,
#                                     step=0.1)))
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam'])
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     if optimizer == 'adam':
#         optimizer = Adam(learning_rate=hp_learning_rate)
#     elif optimizer == 'rmsprop':
#         optimizer = RMSprop(learning_rate=hp_learning_rate)
#     elif optimizer == 'nadam':
#         optimizer = Nadam(learning_rate=hp_learning_rate)
#     else:
#         raise
#     model.compile(optimizer=optimizer,
#                   loss='binary_crossentropy',
#                   metrics='binary_accuracy')
#     return model


dict_len = 20000
string_len = 100
batch_size = 64
epochs = 300
max_trials = 50
test_size = 0.2
validation_split = 0.2
best_model_monitor = 'val_accuracy'
best_model_mode = 'max'
count_best_model = 10
stop_early = EarlyStopping(monitor='val_binary_accuracy',
                           patience=50,  # кол-во эпох без улучшений и стоп
                           min_delta=0.01,  # мин знач ниже которого не будет считаться улучшением
                           verbose=1,  # показывать-1 статистику или нет-0
                           mode='max',  # auto min max понижать или повышать результаты нужно
                           baseline=None,  # прекратит обучение если не достигнет базового уровня
                           restore_best_weights=True)

data = pd.read_csv('data/avoska_morfy.csv', header=None, names=['category', 'text'])
print('Начальный размер базы данных - ', len(data.index))


# Создаем словарь токенайзера со всех текстов размером dict_len слов
# tokenizer = Tokenizer(num_words=dict_len - 1)
tokenizer = Tokenizer(num_words=dict_len)
tokenizer.fit_on_texts(data['text'])
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Обрабатываем данные, оставляем только слова из первых XX.000 слов словаря токенайзера
data['text'] = tokenizer.texts_to_sequences(data['text'])
data['text'] = tokenizer.sequences_to_texts(data['text'])
data = data.drop_duplicates('text', keep='last')
data = data[data['text'].str.strip().astype(bool)]

# Откладываем для теста ВСЕ данные сырые оттокенизированные
x_data_all = data['text'].tolist()
y_data_all = np.asarray(data['category'].tolist())
x_data_all = tokenizer.texts_to_sequences(x_data_all)
x_data_all = pad_sequences(x_data_all, maxlen=string_len)

# Выравнивание данных по ответам
nul_data = data.loc[data['category'] == 0]
odin_data = data.loc[data['category'] == 1]
# Отрезаем от положительных ответов концовку с запасом 1000,
# чтоб тренировочных данных было хоть немного побольше
index = len(odin_data.index) - len(nul_data.index) - 1000
odin_data = odin_data[index:]
data = pd.concat([odin_data, nul_data], ignore_index=True)
# data = data.sample(frac=1).reset_index(drop=True)
# и снова все токенизируем и обрезаем строки
x_data = nul_data['text'].tolist() + odin_data['text'].tolist()
y_data = np.asarray(nul_data['category'].tolist() + odin_data['category'].tolist())
x_data = tokenizer.texts_to_sequences(x_data)
x_data = pad_sequences(x_data, maxlen=string_len)
print('Конечный размер базы данных после токенайзера и очистки - ', x_data.shape, y_data.shape)

# y_data = np_utils.to_categorical(y_data, 2)
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=3,
                                                    shuffle=True)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Создаем словарь-матрицу вложений векторов размером словаря токенайзера
# word2vec_path = '../../data/rusvectores_model.bin'
# word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#
# embedding_matrix = np.zeros((dict_len, 300))
# for word, i in tokenizer.word_index.items():
#     if i == dict_len:
#         break
#     embedding_matrix[i, :] = normalize(word2vec[word], -1)
# print(embedding_matrix.shape)

tuner = kt.BayesianOptimization(model_builder,
                                objective='val_binary_accuracy',
                                directory='test_directory',
                                max_trials=max_trials,
                                executions_per_trial=1,
                                project_name='intro_to_kt')

tuner.search(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=[x_test, y_test],
             callbacks=[stop_early])

tuner.results_summary()

best_models = tuner.get_best_models(count_best_model)
for idx, best_model in enumerate(best_models):
    best_model.summary()
    best_model.evaluate(x_data_all, y_data_all)
    best_model.save('best_model' + str(idx) + '.h5')
