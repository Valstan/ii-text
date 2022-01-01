import gensim
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def udpipe_to_vectors(count_words, test_size, folder_patch):
    model_path = folder_patch + 'rusvectores_model.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    data = pd.read_csv(folder_patch + 'avoska_udpipe.csv', header=None, names=['category', 'text'])
    old_count_data = len(data.index)
    list_category = data['category'].to_list()
    y = np.asarray(list_category)
    list_text_data = data['text'].to_list()

    count = 0
    vectors = None
    for string_udpipe in list_text_data:
        if count % 100 == 0:
            print(count)
        sentence = string_udpipe.split()

        sentence_vec = None
        for word in sentence:
            word_vec = np.expand_dims(model[word], axis=0)
            if sentence_vec is None:
                sentence_vec = word_vec
            else:
                sentence_vec = np.concatenate((sentence_vec, word_vec))

        r, c = np.shape(sentence_vec)
        if r < count_words:
            zeros = np.zeros((count_words - r, 300))
            sentence_vec = np.concatenate((sentence_vec, zeros))
        elif r > count_words:
            sentence_vec = np.split(sentence_vec, [count_words])[0]

        if vectors is None:
            vectors = np.expand_dims(sentence_vec, axis=0)
        else:
            sentence_vec = np.expand_dims(sentence_vec, axis=0)
            vectors = np.concatenate((vectors, sentence_vec))
        count += 1

    print('Было в UDPIPE - ', old_count_data)
    print('Стало - ', vectors.shape)

    x_train, x_test, y_train, y_test = train_test_split(vectors, y, test_size=test_size)

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    print('Тренировочных векторов - ', x_train.shape)
    print('Проверочных векторов - ', x_test.shape)

    np.save(folder_patch + 'x_train', x_train)
    np.save(folder_patch + 'x_test', x_test)
    np.save(folder_patch + 'y_train', y_train)
    np.save(folder_patch + 'y_test', y_test)


if __name__ == '__main__':

    count_w = 50  # Какое количество слов должно быть в примере
    test_sz = 0.2  # Размер тестовых данных
    folder_ptch = '../../data/'  # Путь до директории с данными
    udpipe_to_vectors(count_w, test_sz, folder_ptch)
