import os
import pickle

# import gensim
import keras_tuner as kt
import numpy as np
import pandas as pd
from navec import Navec
# from keras.utils import np_utils
# from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense, Dropout, Embedding
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
    model.add(Embedding(input_dim=dict_len,
                        output_dim=embed,
                        input_length=string_len,
                        mask_zero=True,
                        weights=[embedding_matrix],
                        trainable=True))
    model.add(Dropout(rate=hp.Float('dropout1',
                                    max_value=0.3,
                                    min_value=0.1,
                                    step=0.1)
                      ))
    model.add(GRU(units=hp.Int('gru1units',
                               max_value=64,
                               min_value=8,
                               step=8),
                  activation=activation_choice,
                  dropout=hp.Float('gru1drop',
                                   max_value=0.3,
                                   min_value=0.1,
                                   step=0.1)
                  ))
    model.add(Dropout(rate=hp.Float('dropout2',
                                    max_value=0.3,
                                    min_value=0.1,
                                    step=0.1)
                      ))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam'])
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2])
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
#     model.add(Embedding(input_dim=dict_len + 1,
#                         output_dim=64,
#                         mask_zero=True))
#     model.add(Dropout(rate=hp.Float('dropout1',
#                                     max_value=0.8,
#                                     min_value=0.5,
#                                     step=0.1)
#                       ))
#     model.add(GRU(units=hp.Int('gru1units',
#                                max_value=8,
#                                min_value=2,
#                                step=2),
#                   activation=activation_choice,
#                   dropout=hp.Float('gru1drop',
#                                    max_value=0.8,
#                                    min_value=0.5,
#                                    step=0.1)
#                   ))
#     model.add(Dropout(rate=hp.Float('dropout1',
#                                     max_value=0.8,
#                                     min_value=0.5,
#                                     step=0.1)
#                       ))
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop', 'nadam'])
#     # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2])
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


dict_len = 5000
string_len = 50
batch_size = 8
epochs = 300
max_trials = 10
count_best_model = 10
embed = 300
stop_early = EarlyStopping(monitor='val_loss',
                           patience=5,  # кол-во эпох без улучшений и стоп
                           min_delta=0.01,  # мин знач ниже которого не будет считаться улучшением
                           verbose=1,  # показывать-1 статистику или нет-0
                           mode='min',  # auto min max понижать или повышать результаты нужно
                           baseline=0.33,  # прекратит обучение если не достигнет базового уровня
                           restore_best_weights=False)

data = pd.read_csv('data/avoska_lemms.csv', header=None, names=['category', 'text'])
print('Начальный размер базы данных - ', len(data.index))

print('Выравнивание данных по ответам')
nul_data = data.loc[data['category'] == 0]
print('Количество данных с нулевой оценкой - ', len(nul_data.index))
for cn in range(2):
    data = pd.concat([data, nul_data], ignore_index=True)
    # data = data.sample(frac=1).reset_index(drop=True)
    # data = data.sample(frac=1).reset_index(drop=True)
    # data = data.sample(frac=1).reset_index(drop=True)

print('Создаем словарь токенайзера со всех текстов размером dict_len слов')
tokenizer = Tokenizer(num_words=dict_len - 1)
tokenizer.fit_on_texts(data['text'])
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Готовим данные для обучения
x_train = data['text'].tolist()
y_train = np.asarray(data['category'].tolist())
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=string_len)
print('Данные для обучения - ', x_train.shape, y_train.shape)

# Загружаем данные для валидации, убираем в нем дубликаты мешающие оценке модели
valid = pd.read_csv('data/avoska_test_lemms.csv', header=None, names=['category', 'text'])
valid = valid.drop_duplicates('text', keep='last')
x_valid = valid['text'].tolist()
y_valid = np.asarray(valid['category'].tolist())
x_valid = tokenizer.texts_to_sequences(x_valid)
x_valid = pad_sequences(x_valid, maxlen=string_len)
print('Данные для валидации - ', x_valid.shape, y_valid.shape)

# Загружаем данные для конечного теста, убираем в нем дубликаты мешающие оценке модели
test = pd.read_csv('data/avoska_test_lemms.csv', header=None, names=['category', 'text'])
test = test.drop_duplicates('text', keep='last')
x_test = test['text'].tolist()
y_test = np.asarray(test['category'].tolist())
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=string_len)
print('Данные для тестирования - ', x_test.shape, y_test.shape)

# Создаем словарь-матрицу вложений векторов размером словаря токенайзера
navec_path = 'data/navec_news_v1_1B_250K_300d_100q.tar'
navec = Navec.load(navec_path)

embedding_matrix = np.zeros((dict_len, embed))
for word, i in tokenizer.word_index.items():
    if i == dict_len:
        break
    if word in navec:
        embedding_matrix[i, :] = normalize(navec[word][:embed], -1)
print(embedding_matrix.shape)

tuner = kt.BayesianOptimization(model_builder,
                                objective='val_binary_accuracy',
                                directory='test_directory',
                                max_trials=max_trials,
                                executions_per_trial=1,
                                project_name='intro_to_kt')

tuner.search(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=[x_valid, y_valid],
             callbacks=[stop_early])

tuner.results_summary()

best_models = tuner.get_best_models(count_best_model)

for idx, best_model in enumerate(best_models):
    best_model.summary()
    best_model.evaluate(x_test, y_test)
    best_model.save('best_model' + str(idx) + '.h5')

# Обратно в текст
# data['text'] = tokenizer.sequences_to_texts(data['text'])
# Удаляем дубликаты оставляем одну последнюю версию из них
# data = data.drop_duplicates('text', keep='last')
# Удаляем пустые строки ... вроде бы)))
# data = data[data['text'].str.strip().astype(bool)]

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

# y_data = np_utils.to_categorical(y_data, 2)
# x_train, x_test, y_train, y_test = train_test_split(x_data,
#                                                     y_data,
#                                                     test_size=test_size,
#                                                     random_state=3,
#                                                     shuffle=True)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
