import re

import pandas as pd
import pymorphy2
from pytz import unicode

ma = pymorphy2.MorphAnalyzer()


old_data = pd.read_csv('data/avoska_ai.csv', header=None, names=['category', 'text'])
old_count_human = len(old_data.index)
list_category = old_data['category'].to_list()
list_text = old_data['text'].to_list()

new_data = pd.read_csv('data/avoska_ai_new.csv', header=None, names=['category', 'text'])
old_count_ai = len(new_data.index)

for idx, val in enumerate(list_text):
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

    if text:
        new_data.loc[len(new_data.index)] = [list_category[idx], text]

train = new_data.drop_duplicates('text', keep='last')
new_count = len(train.index)
print('Новых данных сырых - ', old_count_human)
print('Было данных для нейронки - ', old_count_ai)
print('Стало данных для нейронки - ', new_count)

train.to_csv('data/avoska_ai_new.csv', header=False, encoding='utf-8', index=False)
