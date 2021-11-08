import pandas as pd

from get_posts import get_posts

posts = get_posts(50)

train = pd.read_csv('avoska_txt.csv', header=None, names=['category', 'text'])
len_old_train = len(train.index)
list_train_old = train['text'].to_list()
new_texts = []

for i in posts:
    if i['text'] not in list_train_old and i['text'] not in new_texts:
        print(i['text'])
        a = int(input("0 or 1 - "))
        train.loc[len(train.index)] = [a, i['text']]
        new_texts.append(i['text'])

print("Было - ", len_old_train, " записей.")
print("Стало - ", len(train.index), " записей.")

train.to_csv('avoska_txt.csv', header=False, encoding='utf-8', index=False)
