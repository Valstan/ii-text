import pandas as pd

from bin.ai.groups import all_groups
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

lip = pd.read_csv('../../data/lip_grab_posts.csv', header=None, names=['lip'])
lip_lst = lip['lip'].to_list()
sort_posts = []
for i in new_posts:
    id_post = str(i['owner_id']) + '_' + str(i['id'])
    if id_post not in lip_lst:
        lip.loc[len(lip.index)] = id_post
        sort_posts.append(i)

lip.to_csv('../../data/lip_grab_posts.csv', header=False, encoding='utf-8', index=False)

texts = []
for i in sort_posts:
    if 'text' in i and i['text']:
        texts.append(i['text'])

summa = len(texts)

grab = pd.read_csv('../../data/grab_text.csv', header=None, names=['text'])
len_old_grab = len(grab.index)
list_grab_old = grab['text'].to_list()

count = 0
new_texts = []

for i in texts:
    if i not in list_grab_old and i not in new_texts:
        grab.loc[len(grab.index)] = i
        new_texts.append(i)
        grab.to_csv('../../data/temp_grab.csv', header=False, encoding='utf-8', index=False)

grab = grab.drop_duplicates('text', keep='last')
print("\033[3;30;42m  Было:\033[0;0m", len_old_grab)
print("\033[3;30;42m  Стало:\033[0;0m", len(grab.index))

grab.to_csv('../../data/grab_text.csv', header=False, encoding='utf-8', index=False)
