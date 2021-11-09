import json
import os
import time

from bin.rw.change_lp import change_lp
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.utils.clear_copy_history import clear_copy_history

zanachka = {
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
    }}


def get_posts(offset, count):
    session = {
        "name_base": "mi",
        "name_session": "novost",
        "groups": {
            "Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761
        }}
    with open(os.path.join('logpass.json'), 'r', encoding='utf-8') as f:
        lp = json.load(f)
    session.update(lp)
    vkapp = get_session_vk_api(change_lp(session))
    count = count // len(session['groups'])
    posts = []
    for group in session['groups'].values():
        posts += vkapp.wall.get(owner_id=group, count=count, offset=offset, v=5.102)['items']
        time.sleep(1)

    new_posts = []
    for i in posts:
        a = clear_copy_history(i)
        if 'text' in a:
            new_posts.append(a)

    return new_posts
