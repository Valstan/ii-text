import json
import os

import pandas as pd
import pymorphy2

from bin.rw.change_lp import change_lp
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from clean_text import clean_text


def aitext():
    session = {"groups": {
            "Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
            "Обо всем Малмыж https://vk.com/malpod": -89083141,
            "Иван Малмыж https://vk.com/malmyzh.prodam": 364752344,
            "Почитай Малмыж https://vk.com/baraholkaml": 624118736,
            "Первый Малмыжский https://vk.com/malmiz": -86517261}}
    with open(os.path.join('logpass.json'), 'r', encoding='utf-8') as f:
        lp = json.load(f)
    session.update(lp)
    vkapp = get_session_vk_api(change_lp(session))

    new_posts = read_posts(vkapp, session['groups'], 20)

    morph = pymorphy2.MorphAnalyzer(lang='ru')
    stmt = "SELECT * FROM {database}.{table}".format(database=db['database'], table=db['table'])

    df = pd.read_sql(stmt, conn)

    df['Description'] = df.apply(lambda x: clean_text(x[u'Описание заявки']), axis=1)
