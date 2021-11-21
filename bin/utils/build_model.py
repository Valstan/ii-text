from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential


def build_model(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
    model.add(Embedding(10000, 64, input_length=50))
    model.add(Conv1D(250, 7, padding='valid', activation=activation_choice))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=hp.Int('units_input',  # Полносвязный слой с разным количеством нейронов
                                 min_value=512,  # минимальное количество нейронов
                                 max_value=1024,  # максимальное количество
                                 step=32),
                    activation=activation_choice))
    model.add(Dense(units=hp.Int('units_hidden1',
                                 min_value=128,
                                 max_value=512,
                                 step=32),
                    activation=activation_choice))
    model.add(Dropout(rate=hp.Float('rate1',
                                    min_value=0.2,
                                    max_value=0.5,
                                    step=0.05)))
    model.add(Dense(units=hp.Int('units_hidden2',
                                 min_value=64,
                                 max_value=128,
                                 step=16),
                    activation=activation_choice))
    model.add(Dropout(rate=hp.Float('rate2',
                                    min_value=0.1,
                                    max_value=0.3,
                                    step=0.05)))
    model.add(Dense(units=hp.Int('units_hidden3',
                                 min_value=16,
                                 max_value=64,
                                 step=8),
                    activation=activation_choice))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
