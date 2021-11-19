import pandas as pd

from bin.load_txt_to_csv.get_txt_from_posts import get_txt_from_posts


def load_to_human(new_posts):
    lip = pd.read_csv('data/lip_posts.csv', header=None, names=['lip'])
    lip_lst = lip['lip'].to_list()
    sort_posts = []
    for i in new_posts:
        id_post = str(i['owner_id']) + '_' + str(i['id'])
        if id_post not in lip_lst:
            lip.loc[len(lip.index)] = id_post
            sort_posts.append(i)

    lip.to_csv('data/lip_posts.csv', header=False, encoding='utf-8', index=False)

    texts = get_txt_from_posts(sort_posts)

    summa = len(texts)

    train = pd.read_csv('data/avoska_human.csv', header=None, names=['category', 'text'])
    new_train = pd.read_csv('data/new_human_null.csv', header=None, names=['category', 'text'])
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
            new_train.loc[len(new_train.index)] = [a, i]
            new_texts.append(i)

    print("Было - ", len_old_train, " записей.")
    print("Стало - ", len(train.index), " записей.")

    train.to_csv('data/avoska_human.csv', header=False, encoding='utf-8', index=False)
    new_train.to_csv('data/new_human.csv', header=False, encoding='utf-8', index=False)
