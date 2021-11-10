import pandas as pd

from bin.load_txt_to_csv.get_txt_from_posts import get_txt_from_posts
from bin.rw.change_lp import change_lp
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table


def main(count, group, path_file):
    session = get_session('mi', 'config', 'novost')
    session = load_table(session, session['name_session'])
    session['groups'] = group
    vk_app = get_session_vk_api(change_lp(session))

    offset = 0
    posts = []
    while offset <= count-100:
        posts.append(read_posts(vk_app, session['groups'], 100, offset))
        offset += 100

    new_posts = []
    for i in posts:
        new_posts.append(clear_copy_history(i))
    texts = get_txt_from_posts(new_posts)

    train = pd.read_csv(path_file, header=None, names=['text'])
    new_texts = []
    for i in texts:
        train.loc[len(train.index)] = [i]
        new_texts.append(i)

    train.to_csv(path_file, header=False, encoding='utf-8', index=False)


if __name__ == '__main__':
    one = 300  # сколько постов скачать с каждой группы
    two = {"Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
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
           "wizard https://vk.com/public95775916": -95775916}
    three = 'avoska_txt_norating.csv'
    main(one, two, three)
