from bin.rw.change_lp import change_lp
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.sort.human_to_ai import human_to_ai
from bin.rw.load_to_human import load_to_human
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table


groups = dict()
groups["Подслушано Малмыж https://vk.com/podslyshanomalmyj"] = -149841761
groups["Обо всем Малмыж https://vk.com/malpod"] = -89083141
groups["Иван Малмыж https://vk.com/malmyzh.prodam"] = 364752344
groups["Почитай Малмыж https://vk.com/baraholkaml"] = 624118736
groups["Первый Малмыжский https://vk.com/malmiz"] = -86517261
groups["Смешное видео"] = -132265
# groups["МУЗЫКА. МОТОР! Русские видеоклипы https://vk.com/russianmusicvideo"] = -37343149
# groups["МУЗЫКА НУЛЕВЫХ СССР (СУПЕРДИСКОТЕКА 90х - 2000х) https://vk.com/public50638629"] = -50638629
# groups["Музыка 70-х 80-х 90-х 2000-х.Саундтреки ! https://vk.com/public187135362"] = -187135362
# groups["Культура & Искусство Журнал для умных и творческих https://vk.com/public31786047"] = -31786047
# groups["Удивительный мир https://vk.com/ourmagicalworld"] = -42320333
# groups["Случайный Ренессанс Искусство повсюду https://vk.com/accidental_renaissance"] = -92583139
# groups["wizard https://vk.com/public95775916"] = -95775916

count_posts = 100  # сколько постов скачать
offset = 0  # смещение от начала ленты

session = get_session('mi', 'config', 'test')
session = load_table(session, session['name_session'])
session['groups'] = groups
vk_app = get_session_vk_api(change_lp(session))

posts = read_posts(vk_app, session['groups'], count_posts, offset)

new_posts = []
for i in posts:
    new_posts.append(clear_copy_history(i))

load_to_human(new_posts)
human_to_ai()
