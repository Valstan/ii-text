import pandas as pd
import pymorphy2
from pytz import unicode

# Укажите путь до файлов
filepatch_text = '../../data/avoska_human.csv'
filepatch_morfy = '../../data/avoska_morfy.csv'

ma = pymorphy2.MorphAnalyzer()

text = pd.read_csv(filepatch_text, header=None, names=['category', 'text'])
old_count_text = len(text.index)
list_category = text['category'].to_list()
list_text = text['text'].to_list()

morfy = pd.read_csv(filepatch_morfy, header=None, names=['category', 'text'])
old_count_morfy = len(morfy.index)
morfy = morfy.iloc[0:0]

for idx, text in enumerate(list_text):
    if idx % 100 == 0:
        print(idx)
    text = text.lower()
    text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 2)
    if text:
        morfy.loc[len(morfy.index)] = [list_category[idx], text]

morfy = morfy.drop_duplicates('text', keep='last')
new_count = len(morfy.index)
print('Было данных morfy - ', old_count_morfy)
print('Новых данных text - ', old_count_text)
print('Стало данных morfy - ', new_count)

morfy.to_csv(filepatch_morfy, header=False, encoding='utf-8', index=False)
