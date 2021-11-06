import json
import os

from bin.rw.change_lp import change_lp
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history


def get_posts(count):

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
    with open(os.path.join('logpass.json'), 'r', encoding='utf-8') as f:
        lp = json.load(f)
    session.update(lp)
    vkapp = get_session_vk_api(change_lp(session))
    count = count // len(session['groups'])
    posts = read_posts(vkapp, session['groups'], 5)
    new_posts = []
    for i in posts:
        new_posts.append(clear_copy_history(i))

    return new_posts
