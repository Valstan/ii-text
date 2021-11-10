import pandas as pd

from bin.rw.change_lp import change_lp
from bin.rw.get_image import image_get
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.driver import load_table
from free_ocr import free_ocr

session = get_session('mi', 'config', 'novost')
session = load_table(session, session['name_session'])
session['groups'] = {"Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761}
vkapp = get_session_vk_api(change_lp(session))
count = 100
offset = 0
posts = read_posts(vkapp, session['groups'], count, offset)

texts = []
count_pictures = 0
for i in posts:
    if i['text']:
        texts.append(i['text'])
    if 'attachments' in i and i['attachments'][0]['type'] == 'photo':
        for x in i['attachments'][0]['photo']['sizes']:
            height = 0
            url = ''
            if x['height'] > height:
                height = x['height']
                url = x['url']
        if image_get(url, 'image'):
            a = free_ocr('image')
            count_pictures += 1
            print('Картинок распознано - ', count_pictures)
            if a:
                texts.append(a)

train = pd.read_csv('avoska_txt.csv', header=None, names=['category', 'text'])
len_old_train = len(train.index)
list_train_old = train['text'].to_list()

new_texts = []
for i in texts:
    if i not in list_train_old and i not in new_texts:
        print(i)
        a = int(input("0 or 1 - "))
        train.loc[len(train.index)] = [a, i]
        new_texts.append(i)

print("Было - ", len_old_train, " записей.")
print("Стало - ", len(train.index), " записей.")

train.to_csv('avoska_txt.csv', header=False, encoding='utf-8', index=False)

'''zanachka = {
    "name_base": "mi",
    "name_session": "novost",
    "groups": {
        "Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
        "Обо всем Малмыж https://vk.com/malpod": -89083141,
        "Иван Малмыж https://vk.com/malmyzh.prodam": 364752344,
        "Почитай Малмыж https://vk.com/baraholkaml": 624118736,
        "Первый Малмыжский https://vk.com/malmiz": -86517261,
        "Смешное видео": -132265,
        "МУЗЫКА. МОТОР! Русские видеоклипы https://vk.com/russianmusicvideo": -37343149,
        "МУЗЫКА НУЛЕВЫХ СССР (СУПЕРДИСКОТЕКА 90х - 2000х) https://vk.com/public50638629": -50638629,
        "Музыка 70-х 80-х 90-х 2000-х.Саундтреки ! https://vk.com/public187135362": -187135362,
        "Культура & Искусство Журнал для умных и творческих https://vk.com/public31786047": -31786047,
        "Удивительный мир https://vk.com/ourmagicalworld": -42320333,
        "Случайный Ренессанс Искусство повсюду https://vk.com/accidental_renaissance": -92583139,
        "wizard https://vk.com/public95775916": -95775916
    }}'''