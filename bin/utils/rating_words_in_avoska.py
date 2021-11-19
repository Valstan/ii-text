import json
import os
from collections import Counter

import pandas as pd

train = pd.read_csv("../../data/avoska_ai.csv", header=None, names=['category', 'text'])
list_train = train['text'].to_list()

list_train_new = []
for i in list_train:
    d = str(i)
    d = list(set(d.split()))
    list_train_new += d

rating_words = dict(Counter(list_train_new))
rating_words = {k: v for k, v in sorted(rating_words.items(), key=lambda item: item[1], reverse=True)}

with open(os.path.join('../../data/rating_words_in_avoska.json'),
          'w', encoding='utf-8') as f:
    f.write(json.dumps(rating_words, indent=2, ensure_ascii=False))
