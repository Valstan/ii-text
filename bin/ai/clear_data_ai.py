import pandas as pd
import pymorphy2
from pytz import unicode

ma = pymorphy2.MorphAnalyzer()


old_data = pd.read_csv('../../data/avoska_test_human.csv', header=None, names=['category', 'text'])
old_count_human = len(old_data.index)
list_category = old_data['category'].to_list()
list_text = old_data['text'].to_list()

new_data = pd.read_csv('../../data/avoska_test_morfy.csv', header=None, names=['category', 'text'])
old_count_ai = len(new_data.index)

for idx, text in enumerate(list_text):
    if idx % 100 == 0:
        print(idx)
    text = text.lower()
    text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 2)
    if text:
        new_data.loc[len(new_data.index)] = [list_category[idx], text]

train = new_data.drop_duplicates('text', keep='last')
new_count = len(train.index)
print('Новых данных сырых - ', old_count_human)
print('Было данных для нейронки - ', old_count_ai)
print('Стало данных для нейронки - ', new_count)

train.to_csv('../../data/avoska_test_morfy.csv', header=False, encoding='utf-8', index=False)
