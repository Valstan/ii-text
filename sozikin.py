from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


num_words = 10000
max_review_len = 100
train = pd.read_csv('avoska_txt.csv', header=None, names=['Class', 'Review'])
reviews = train['Review']
y_train = train['Class']
tokenizer = Tokenizer(num_words=num_words)
print("Первый токинайзер")
tokenizer.fit_on_texts(reviews)
print("Второй токинайзер")
sequences = tokenizer.texts_to_sequences(reviews)
print("Токинайзеры завершены")
x_train = pad_sequences(sequences, maxlen=max_review_len)

model = Sequential()
model.add(Embedding(num_words, 64, input_length=max_review_len))
model.add(Conv1D(250, 5, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model_save_path = "best_model.h5"
checkpoint_callback = ModelCheckpoint(model_save_path,
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)

print("Обучаем модель")
history = model.fit(x_train,
                    y_train,
                    epochs=4,
                    batch_size=128,
                    validation_split=0.1,
                    callbacks=[checkpoint_callback])

plt.plot(history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
# model = load_model('my_model.h5')

print("pause")
print("stop")
