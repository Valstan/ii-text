import pickle

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


dict_max_words = 3000  # Размер словаря слов

baza = pd.read_csv('data/avoska_udpipe.csv', header=None, names=['category', 'text'])

tokenizer = Tokenizer(num_words=dict_max_words, filters='', lower=False)
tokenizer.fit_on_texts(baza['text'])
with open('data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('В словаре {} слов'.format(len(tokenizer.word_index)))

baza['text'] = tokenizer.texts_to_sequences(baza['text'])
baza['text'] = tokenizer.sequences_to_texts(baza['text'])
baza = baza.drop_duplicates('text', keep='last')
baza = baza[baza['text'].str.strip().astype(bool)]

baza.to_csv('data/avoska_udpipe_dict.csv', header=False, encoding='utf-8', index=False)