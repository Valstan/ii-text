from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.models import Sequential


def lstm_drop_dense_drop(hp):
    model = Sequential()
    activation_choice = hp.Choice('activation', values=['relu', 'tanh', 'elu', 'selu'])
    model.add(Embedding(input_dim=5000,
                        output_dim=hp.Int('out_dim',
                                          min_value=8,
                                          max_value=64,
                                          step=8),
                        input_length=50))

    model.add(LSTM(units=hp.Int('units',
                                min_value=32,
                                max_value=512,
                                step=32),
                   activation=activation_choice,
                   dropout=hp.Float('drop',
                                    min_value=0.1,
                                    max_value=0.5,
                                    step=0.05),
                   recurrent_dropout=hp.Float('drop_rec',
                                              min_value=0.1,
                                              max_value=0.3,
                                              step=0.05),
                   ))
    model.add(Dropout(rate=hp.Float('drop1',
                                    min_value=0.1,
                                    max_value=0.5,
                                    step=0.05)))
    model.add(Dense(units=hp.Int('units_hidden',
                                 min_value=32,
                                 max_value=512,
                                 step=32),
                    activation=activation_choice))
    model.add(Dropout(rate=hp.Float('drop2',
                                    min_value=0.1,
                                    max_value=0.3,
                                    step=0.05)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    return model
