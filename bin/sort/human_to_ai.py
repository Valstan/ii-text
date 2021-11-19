import re

import pandas as pd
import pymorphy2
from pytz import unicode


def human_to_ai():
    print('Начинаем обработку текста для нейронки и перенос в авоську')
    ma = pymorphy2.MorphAnalyzer()

    train_human = pd.read_csv('data/new_human.csv', header=None, names=['category', 'text'])
    old_count_human = len(train_human.index)
    list_category = train_human['category'].to_list()
    list_text_human = train_human['text'].to_list()

    train_ai = pd.read_csv('data/avoska_ai.csv', header=None, names=['category', 'text'])
    old_count_ai = len(train_ai.index)
    list_text_ai = train_ai['text'].to_list()

    list_text_new = []
    for idx, val in enumerate(list_text_human):
        if idx % 100 == 0:
            print(idx)
        d = str(val)
        d = re.sub("[^\w]", " ", d)
        d = re.sub("_", " ", d)
        d = d.lower()
        d = " ".join(ma.parse(unicode(word))[0].normal_form for word in d.split())
        d = re.sub(" анон", " ", d)
        d = re.sub("анон ", " ", d)
        d = re.sub("анонимно", " ", d)
        d = re.sub("аноним", " ", d)
        d = ' '.join(word for word in d.split() if len(word) > 3)
        d = list(set(d.split()))
        d = ' '.join(str(e) for e in d)
        if not d or d == '' or d == ' ' or d == '  ' or d in 'ананимноанонимно':
            continue
        if d not in list_text_new and d not in list_text_ai:
            train_ai.loc[len(train_ai.index)] = [list_category[idx], d]
            list_text_new += d

    train = train_ai.drop_duplicates('text', keep='last')
    new_count = len(train.index)
    print('Новых данных сырых - ', old_count_human)
    print('Было данных для нейронки - ', old_count_ai)
    print('Стало данных для нейронки - ', new_count)

    train.to_csv('data/avoska_ai.csv', header=False, encoding='utf-8', index=False)
