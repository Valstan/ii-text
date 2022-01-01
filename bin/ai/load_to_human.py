import pandas as pd

from bin.load_txt_to_csv.get_txt_from_posts import get_txt_from_posts
from bin.rw.change_lp import change_lp
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table

groups = dict()
groups["Подслушано Малмыж https://vk.com/podslyshanomalmyj"] = -149841761
groups["Обо всем Малмыж https://vk.com/malpod"] = -89083141
groups["Иван Малмыж https://vk.com/malmyzh.prodam"] = 364752344
groups["Почитай Малмыж https://vk.com/baraholkaml"] = 624118736
groups["Первый Малмыжский https://vk.com/malmiz"] = -86517261
groups["Смешное видео"] = -132265
groups["МУЗЫКА. МОТОР! Русские видеоклипы https://vk.com/russianmusicvideo"] = -37343149
groups["МУЗЫКА НУЛЕВЫХ СССР (СУПЕРДИСКОТЕКА 90х - 2000х) https://vk.com/public50638629"] = -50638629
groups["Музыка 70-х 80-х 90-х 2000-х.Саундтреки ! https://vk.com/public187135362"] = -187135362
groups["Культура & Искусство Журнал для умных и творческих https://vk.com/public31786047"] = -31786047
groups["Удивительный мир https://vk.com/ourmagicalworld"] = -42320333
groups["Случайный Ренессанс Искусство повсюду https://vk.com/accidental_renaissance"] = -92583139
groups["wizard https://vk.com/public95775916"] = -95775916

count_posts = 100  # сколько постов скачать
offset = 0  # смещение от начала ленты

session = get_session('dran', 'config', 'novost', '../../logpass.json')
session = load_table(session, session['name_session'])
session['groups'] = groups
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

train = pd.read_csv('../../data/avoska_human.csv', header=None, names=['category', 'text'])
len_old_train = len(train.index)
list_train_old = train['text'].to_list()

count = 0
new_texts = []
for i in texts:
    count += 1
    if i not in list_train_old and i not in new_texts:
        print(i)
        a = int(input())
        train.loc[len(train.index)] = [a, i]
        new_texts.append(i)
    print(f"****** {summa}-{count} ***************")

train = train.drop_duplicates('text', keep='last')
print("Было - ", len_old_train, " записей.")
print("Стало - ", len(train.index), " записей.")

train.to_csv('../../data/avoska_human.csv', header=False, encoding='utf-8', index=False)
