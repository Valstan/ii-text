import json
import os

from keras.preprocessing.text import Tokenizer


def tokenizer_words(txt_list):
    with open(os.path.join('tokenizer.json'), 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)
    if not tokenizer:
        tokenizer = Tokenizer()
    tokenizer.fit_on_texts(txt_list)
    text_sequences = tokenizer.texts_to_sequences(txt_list)

    with open(os.path.join('tokenizer.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer, indent=2, ensure_ascii=False))

    return text_sequences
