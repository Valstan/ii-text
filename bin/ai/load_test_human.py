import pandas as pd

from bin.ai.groups import all_groups
from bin.load_txt_to_csv.get_txt_from_posts import get_txt_from_posts
from bin.rw.change_lp import change_lp
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table


count_posts = 100  # сколько постов скачать
offset = 0  # смещение от начала ленты

session = get_session('dran', 'config', 'novost', '../../logpass.json')
session = load_table(session, session['name_session'])
session['groups'] = all_groups
vk_app = get_session_vk_api(change_lp(session))

posts = read_posts(vk_app, session['groups'], count_posts, offset)

new_posts = []
for i in posts:
    new_posts.append(clear_copy_history(i))

lip = pd.read_csv('../../data/lip_posts.csv', header=None, names=['lip'])
lip_lst = lip['lip'].to_list()
sort_posts = []
for i in new_posts:
    id_post = str(i['owner_id']) + '_' + str(i['id'])
    if id_post not in lip_lst:
        lip.loc[len(lip.index)] = id_post
        sort_posts.append(i)

lip.to_csv('../../data/lip_posts.csv', header=False, encoding='utf-8', index=False)

texts = get_txt_from_posts(sort_posts)

summa = len(texts)

human = pd.read_csv('../../data/avoska_human.csv', header=None, names=['category', 'text'])
len_old_human = len(human.index)
list_human_old = human['text'].to_list()
test = pd.read_csv('../../data/avoska_test_human.csv', header=None, names=['category', 'text'])
len_old_test = len(test.index)
list_test_old = test['text'].to_list()

count = 0
new_texts = []

for i in texts:
    count += 1
    if i not in list_human_old and i not in list_test_old and i not in new_texts:
        a = 2
        print(i)
        while True:
            try:
                a = int(input())
            except:
                print('\033[3;31mКакая-то ошибка!!! Введи цифру 1 или 0\033[0;0m')
            if a == 0 or a == 1:
                break
            else:
                print('\033[3;31mВведите 0 или 1, а вы ввели - \033[0;0m', a)

        test.loc[len(test.index)] = [a, i]
        new_texts.append(i)
    else:
        dubl_test = test.loc[test['text'] == i]
        dubl_human = human.loc[human['text'] == i]
        dubl = pd.concat([dubl_test, dubl_human], ignore_index=True)
        print(dubl)
        test = pd.concat([test, dubl[:1]], ignore_index=True)
    if count % 30 == 0:
        test.to_csv('../../data/temp.csv', header=False, encoding='utf-8', index=False)
    print(f"\033[3;30;42m    {summa}-{count}     \033[0;0m")

# train = train.drop_duplicates('text', keep='last')
print("\033[3;30;42m  Human:\033[0;0m", len_old_human)
print("\033[3;30;42m  Было test:\033[0;0m", len_old_test)
print("\033[3;30;42m  Стало test:\033[0;0m", len(test.index))

test.to_csv('../../data/avoska_test_human.csv', header=False, encoding='utf-8', index=False)
