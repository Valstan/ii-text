import pandas as pd

from bin.load_txt_to_csv.get_txt_from_posts import get_txt_from_posts
from bin.rw.change_lp import change_lp
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table


def main(count, offset, group, path_file):
    session = get_session('mi', 'config', 'novost')
    session = load_table(session, session['name_session'])
    session['groups'] = group
    vk_app = get_session_vk_api(change_lp(session))
    posts = read_posts(vk_app, session['groups'], count, offset)
    new_posts = []
    for i in posts:
        new_posts.append(clear_copy_history(i))
    texts = get_txt_from_posts(new_posts)
    summa = len(texts)

    train = pd.read_csv(path_file, header=None, names=['category', 'text'])
    len_old_train = len(train.index)
    list_train_old = train['text'].to_list()

    count = 0
    new_texts = []
    for i in texts:
        count += 1
        if i not in list_train_old and i not in new_texts:
            print(i)
            a = int(input(f"{summa}-{count} ... 0 or 1 or 666 - "))
            if a == 666:
                break
            train.loc[len(train.index)] = [a, i]
            new_texts.append(i)

    print("Было - ", len_old_train, " записей.")
    print("Стало - ", len(train.index), " записей.")

    train.to_csv(path_file, header=False, encoding='utf-8', index=False)


if __name__ == '__main__':
    one = 50  # сколько постов скачать
    two = 0  # смещение от начала ленты
    three = {"Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
             "Обо всем Малмыж https://vk.com/malpod": -89083141,
             "Иван Малмыж https://vk.com/malmyzh.prodam": 364752344,
             "Почитай Малмыж https://vk.com/baraholkaml": 624118736,
             "Первый Малмыжский https://vk.com/malmiz": -86517261
             }
    four = 'avoska.csv'
    main(one, two, three, four)
