import pickle

from keras_tuner import BayesianOptimization
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from bin.rw.save_best_models import save_best_models
from bin.utils.load_data_from_arrays import load_data_from_arrays


def bm_dense5_binary(hp):
    model = Sequential()
    activation_choice = 'relu'
    # activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'elu', 'selu'])
    model.add(Embedding(input_dim=5000,
                        output_dim=64,
                        input_length=70))
    model.add(GRU(units=64,
                  activation=activation_choice
                  ))

    model.add(Dense(units=512,
                    activation=activation_choice
                    ))
    model.add(Dropout(0.2))

    model.add(Dense(units=256,
                    activation=activation_choice
                    ))
    model.add(Dropout(0.2))

    model.add(Dense(units=128,
                    activation=activation_choice
                    ))
    model.add(Dropout(0.1))

    model.add(Dense(units=64,
                    activation=activation_choice
                    ))
    model.add(Dense(8,
                    activation=activation_choice
                    ))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='rmsprop',
        # optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


def dense5_binary(all_text, all_cat, dict_max_words, string_max_words, project_name, max_trials, batch_size, epochs):
    tokenizer = Tokenizer(num_words=dict_max_words)
    tokenizer.fit_on_texts(all_text)
    total_words = len(tokenizer.word_index)
    print('В словаре {} слов'.format(total_words))
    # Сохраняем токенайзер
    with open('data/' + project_name + '/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Делим данные на две части
    x_train, y_train, x_test, y_test = load_data_from_arrays(all_text, all_cat, 0.1)

    # Секвенируем текст в цифры
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    # Обрезаем длину строк по максимальному количеству слов
    x_train = pad_sequences(x_train, maxlen=string_max_words)
    x_test = pad_sequences(x_test, maxlen=string_max_words)

    # y_train = np_utils.to_categorical(y_train, 2)
    # y_test = np_utils.to_categorical(y_test, 2)

    tuner = BayesianOptimization(
        bm_dense5_binary,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='test_directory',
        project_name=project_name,
        overwrite=True
    )

    # tuner.search_space_summary()

    tuner.search(x_train,
                 y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_split=0.1
                 )
    tuner.results_summary()

    save_best_models(tuner, x_test, y_test, project_name)
