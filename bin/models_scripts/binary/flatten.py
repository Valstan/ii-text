import pickle

from keras_tuner import BayesianOptimization
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from bin.utils.load_data_from_arrays import load_data_from_arrays


def bm_flatten(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'elu', 'selu'])
    model.add(Embedding(input_dim=5000,
                        output_dim=hp.Int('out_dim',
                                          min_value=8,
                                          max_value=64,
                                          step=8),
                        input_length=50))
    model.add(Dropout(rate=hp.Float('drop1',
                                    min_value=0.1,
                                    max_value=0.3,
                                    step=0.05)))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model


def flatten(all_text, all_cat, dict_max_words, string_max_words, project_name, max_trials, batch_size, epochs):
    tokenizer = Tokenizer(num_words=dict_max_words)
    tokenizer.fit_on_texts(all_text)
    total_words = len(tokenizer.word_index)
    print('В словаре {} слов'.format(total_words))
    # Сохраняем токенайзер
    with open('data/' + project_name + '/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Делим данные на две части
    x_train, y_train, x_test, y_test = load_data_from_arrays(all_text, all_cat, train_test_split=0.8)

    # Секвенируем текст в цифры
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    # Обрезаем длину строк по максимальному количеству слов
    x_train = pad_sequences(x_train, maxlen=string_max_words)
    x_test = pad_sequences(x_test, maxlen=string_max_words)

    tuner = BayesianOptimization(
        bm_flatten,  # функция создания модели
        objective='val_accuracy',  # метрика, которую нужно оптимизировать -
        # доля правильных ответов на проверочном наборе данных
        max_trials=max_trials,  # максимальное количество запусков обучения
        executions_per_trial=1,
        directory='test_directory',  # каталог, куда сохраняются обученные сети
        project_name=project_name,  # в папку с именем проекта
        overwrite=True  # перезаписывать файлы
    )

    # tuner.search_space_summary()

    tuner.search(x_train,  # Данные для обучения
                 y_train,  # Правильные ответы
                 batch_size=batch_size,  # Размер мини-выборки
                 epochs=epochs,  # Количество эпох обучения
                 validation_data=(x_test, y_test)
                 )

    # tuner.results_summary()
    model = tuner.get_best_models()
    model.save('data/' + project_name + "/best_model.h5")
