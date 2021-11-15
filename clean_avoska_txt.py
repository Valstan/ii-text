import re

import pandas as pd
import pymorphy2
from pytz import unicode
from sklearn.utils import shuffle

ma = pymorphy2.MorphAnalyzer()

train = pd.read_csv("avoska_old.csv", header=None, names=['category', 'text'])
old_count = len(train.index)
list_category = train['category'].to_list()
list_text = train['text'].to_list()

train_new = pd.DataFrame(columns=['category', 'text'])

list_text_new = []
for idx, val in enumerate(list_text):
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
    if d not in list_text_new:
        train_new.loc[len(train_new.index)] = [list_category[idx], d]
        list_text_new += d

train = train_new.drop_duplicates('text', keep='last')
train = shuffle(train)
new_count = len(train.index)
print(old_count)
print(new_count)

train.to_csv('avoska.csv', header=False, encoding='utf-8', index=False)
