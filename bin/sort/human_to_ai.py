import re

import pandas as pd
import pymorphy2
from pytz import unicode

ma = pymorphy2.MorphAnalyzer()


def human_to_ai():
    print('Начинаем обработку текста для нейронки и перенос в авоську')

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
        text = str(val)
        text = text.lower()
        text = re.sub(r"(\b|не|не )ан[оа]н\w*|"
                      r"п[оа]жалу?й?ст[ао]|"
                      r"админ[уы]? пр[ао]пустит?е?|"
                      r"админ[уы]?\b|"
                      r"Здрав?с?т?в?у?й?т?е?|"
                      r"[^a-zA-Z0-9а-яА-Я]+",
                      ' ', text)

        text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())

        text = ' '.join(word for word in text.split() if len(word) > 3)
        text = list(set(sorted(text.split())))
        text = ' '.join(str(e) for e in text)

        if text and text not in list_text_new and text not in list_text_ai:
            train_ai.loc[len(train_ai.index)] = [list_category[idx], text]
            list_text_new += text

    train = train_ai.drop_duplicates('text', keep='last')
    new_count = len(train.index)
    print('Новых данных сырых - ', old_count_human)
    print('Было данных для нейронки - ', old_count_ai)
    print('Стало данных для нейронки - ', new_count)

    train.to_csv('data/avoska_ai.csv', header=False, encoding='utf-8', index=False)
