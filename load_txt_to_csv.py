import pandas as pd

from get_posts import get_posts

session = {
    "name_base": "mi",
    "name_session": "novost",
    "groups": {
        "Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
        "Обо всем Малмыж https://vk.com/malpod": -89083141,
        "Иван Малмыж https://vk.com/malmyzh.prodam": 364752344,
        "Почитай Малмыж https://vk.com/baraholkaml": 624118736,
        "Первый Малмыжский https://vk.com/malmiz": -86517261
    }}

posts = get_posts(100)
train = pd.read_csv('train.csv')
list_train_old = train['text'].to_list()
new_texts = []

for i in posts:
    if i['text'] not in list_train_old and i['text'] not in new_texts:
        print(i['text'])
        a = int(input("0 or 1 - "))
        train.loc[len(train.index)] = [a, i['text']]
        new_texts.append(i['text'])

train.to_csv('train.csv', encoding='utf-8', index=False)
