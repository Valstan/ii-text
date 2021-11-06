import pickle

from keras.preprocessing.text import Tokenizer


def tokenizer_words(txt_list):

    tokenizer = Tokenizer()

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenizer.fit_on_texts(txt_list)
    text_sequences = tokenizer.texts_to_sequences(txt_list)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return text_sequences
