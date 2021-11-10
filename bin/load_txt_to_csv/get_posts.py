import json
import os
import time

from bin.rw.change_lp import change_lp
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.utils.clear_copy_history import clear_copy_history


def get_posts(offset, count):
    session = {
        "name_base": "mi",
        "name_session": "novost",
        "groups": {
            "Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761
        }}
    with open(os.path.join('../../logpass.json'), 'r', encoding='utf-8') as f:
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
