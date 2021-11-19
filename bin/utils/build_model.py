from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential


def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Embedding(10000, 64, input_length=50))
    model.add(Conv1D(250, 5, padding='valid', activation=activation_choice))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=hp.Int('units_input',  # Полносвязный слой с разным количеством нейронов
                                 min_value=512,  # минимальное количество нейронов - 128
                                 max_value=1024,  # максимальное количество - 1024
                                 step=32),
                    activation=activation_choice))
    model.add(Dropout(0.2))
    model.add(Dense(units=hp.Int('units_hidden',
                                 min_value=128,
                                 max_value=600,
                                 step=32),
                    activation=activation_choice))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
