import pandas as pd
from kerastuner.tuners import RandomSearch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from bin.utils.build_model import build_model

num_words = 10000
max_review_len = 50
train = pd.read_csv('data/avoska_ai.csv', header=None, names=['Class', 'Review'])
reviews = train['Review']
y_train = train['Class']
tokenizer = Tokenizer(num_words=num_words)
print("Первый токинайзер")
tokenizer.fit_on_texts(reviews)
print("Второй токинайзер")
sequences = tokenizer.texts_to_sequences(reviews)
print("Токинайзеры завершены")
x_train = pad_sequences(sequences, maxlen=max_review_len)

tuner = RandomSearch(
    build_model,  # функция создания модели
    objective='val_accuracy',  # метрика, которую нужно оптимизировать -
    # доля правильных ответов на проверочном наборе данных
    max_trials=80,  # максимальное количество запусков обучения
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
