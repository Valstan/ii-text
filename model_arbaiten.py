import io
import os
import pickle

import pymorphy2
from pytz import unicode
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ma = pymorphy2.MorphAnalyzer()
model = keras.models.load_model("BestModels/85_20_93/best_model0.h5")
with open('BestModels/85_20_93/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

key = 1
while key:
    textin = ''
    with io.open('text_for_model_arbaiten.txt', encoding='utf-8') as file:
        for line in file:
            textin += line
    text = textin.lower()
    text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 2)
    text = tokenizer.texts_to_sequences(text.split())
    text = list(word[0] for word in text if word)
    new_text = list()
    new_text.append(text)
    text = pad_sequences(new_text, maxlen=100)

    result = model.predict(text)[0][0]
    print(textin[:40])
    if result > 0.5:
        print('Да! Этот пост можно печатать. - ', result)
    else:
        print('Нет! Этот пост нельзя печатать. - ', result)
    key = input()
