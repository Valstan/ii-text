import json
import os
import re
from collections import Counter
from sklearn.utils import shuffle

import pandas as pd


train = pd.read_csv("avoska.csv", header=None, names=['category', 'text'])
list_train_old = train['text'].to_list()
list_train_new = []
for i in list_train_old:
    d = str(i)
    d = d.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    d = d.lower()
    d = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', d)  # deleting newlines and line-breaks
    d = ' '.join(word for word in d.split() if len(word) > 3)
    d = list(set(d.split()))
    list_train_new += d

rating_words = dict(Counter(list_train_new))
rating_words = {k: v for k, v in sorted(rating_words.items(), key=lambda item: item[1], reverse=True)}

with open(os.path.join('rating_words_in_avoska.json'),
          'w', encoding='utf-8') as f:
    f.write(json.dumps(rating_words, indent=2, ensure_ascii=False))


'''
len_old_train = len(train.index)

# Перемешать строки в dataframe
train = shuffle(train)

# Убрать дубликаты из dataframe
train = train.drop_duplicates('text', keep='last')

# Сохранить dataframe в файл
train.to_csv("avoska.csv", header=False, encoding='utf-8', index=False)
print(rating_words)'''

