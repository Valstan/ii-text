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
                        output_dim=300,
                        input_length=string_len,
                        mask_zero=True,
                        weights=[embedding_matrix],
                        trainable=True))
    model.add(GRU(units=hp.Int('gru1units',
                               max_value=254,
                               min_value=64,
                               step=64),
                  activation=activation_choice,
                  dropout=hp.Float('gru1drop',
                                   max_value=0.3,
                                   min_value=0.1,
                                   step=0.1),
                  return_sequences=True
                  ))
    model.add(Dropout(rate=hp.Float('dropout1',
                                    max_value=0.3,
                                    min_value=0.1,
                                    step=0.1)))
    model.add(GRU(units=hp.Int('gru2units',
                               max_value=64,
                               min_value=16,
                               step=16),
                  activation=activation_choice,
                  dropout=hp.Float('gru2drop',
                                   max_value=0.2,
                                   min_value=0.1,
                                   step=0.05)
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


dict_len = 5000
string_len = 50
batch_size = 32
epochs = 100
max_trials = 30
test_size = 0.2
validation_split = 0.2
best_model_monitor = 'val_accuracy'
best_model_mode = 'max'
count_best_model = 5
stop_early = EarlyStopping(monitor='val_loss',
                           patience=20,  # кол-во эпох без улучшений и стоп
                           min_delta=0.01,  # мин знач ниже которого не будет считаться улучшением
                           verbose=1,  # показывать-1 статистику или нет-0
                           mode='min',  # auto min max понижать или повышать результаты нужно
                           baseline=None,  # прекратит обучение если не достигнет базового уровня
                           restore_best_weights=False)

data = pd.read_csv('../../data/avoska_udpipe.csv', header=None, names=['category', 'text'])
print('Начальный размер базы данных - ', data.index)

# Создаем словарь токенайзера со всех текстов размером dict_len слов
tokenizer = Tokenizer(num_words=dict_len - 1, filters='', lower=False)
tokenizer.fit_on_texts(data['text'])
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Обрабатываем данные, оставляем только слова из первых 10.000 слов словаря токенайзера
data['text'] = tokenizer.texts_to_sequences(data['text'])
data['text'] = tokenizer.sequences_to_texts(data['text'])
data = data.drop_duplicates('text', keep='last')
data = data[data['text'].str.strip().astype(bool)]
x_data = np.asarray(data['text'].tolist())
y_data = np.asarray(data['category'].tolist())
print('Конечный размер базы данных - ', x_data.shape)

x_data = tokenizer.texts_to_sequences(x_data)
x_data = pad_sequences(x_data, maxlen=string_len)
# y_data = np_utils.to_categorical(y_data, 2)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)

# Создаем словарь-матрицу вложений векторов размером словаря токенайзера
word2vec_path = '../../data/rusvectores_model.bin'
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

embedding_matrix = np.zeros((dict_len, 300))
for word, i in tokenizer.word_index.items():
    if i == dict_len:
        break
    embedding_matrix[i, :] = normalize(word2vec[word], -1)
print(embedding_matrix.shape)

tuner = kt.BayesianOptimization(model_builder,
                                objective='val_loss',
                                directory='test_directory',
                                max_trials=max_trials,
                                executions_per_trial=1,
                                project_name='intro_to_kt')

tuner.search(x_train, y_train,
             epochs=epochs,
             batch_size=batch_size,
             validation_split=validation_split,
             callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

'''print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")'''

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
modela = tuner.hypermodel.build(best_hps)
history = modela.fit(x_train, y_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_loss']
best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(x_train, y_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(x_test, y_test)
print("[test loss, test accuracy]:", eval_result)

hypermodel.save('hypermodel.h5')

tuner.results_summary()

best_models = tuner.get_best_models(count_best_model)
for idx, best_model in enumerate(best_models):
    best_model.summary()
    score = best_model.evaluate(x_test, y_test)
    print('Потери при тестировании: ', score[0])
    print('Точность при тестировании:', score[1])
    best_model.save('best_model' + str(idx) + '.h5')
