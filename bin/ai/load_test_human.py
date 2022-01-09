import pandas as pd

from bin.load_txt_to_csv.get_txt_from_posts import get_txt_from_posts
from bin.rw.change_lp import change_lp
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table

groups = {"Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
          "Обо всем Малмыж https://vk.com/malpod": -89083141,
          "Иван Малмыж https://vk.com/malmyzh.prodam": 364752344,
          "Почитай Малмыж https://vk.com/baraholkaml": 624118736,
          "Первый Малмыжский https://vk.com/malmiz": -86517261,
          "Дом-Культуры Малмыжа РЦКиД Районный Центр Культуры и Досуга https://vk.com/id234960216": 234960216,
          "Администрация Малмыжского городского поселения https://vk.com/gormalm": -159098271,
          "Газета Сельская правда https://vk.com/public179280169": -179280169,
          "Малмыжский Краеведческий-Музей https://vk.com/id288616707": 288616707,
          "Сообщество предпринимателей Малмыжского района https://vk.com/public133732168": -133732168,
          "Малмыжская детская школа искусств https://vk.com/club124138214": -124138214,
          "Малмыж и Малмыжский район. Ищу тебя. Разыскивает https://vk.com/club166452860": -166452860,
          "Вятка https://vk.com/vyatkakirovtockaru": -84767981,
          "Добровольцы г. Малмыж Создаём Добрый Малмыж https://vk.com/mvolonter": -72660310,
          "Новости Малмыж https://vk.com/club120893935": -120893935,
          "За мост через Вятку https://vk.com/club111892671": -111892671,
          "Малышок-онлайн Детский сад https://vk.com/malyshok.online": -197351557,
          "Малмыжский завод по ремонту дизельных двигателей https://vk.com/rmz43": -195583920,
          "Культура,молодежная политика и спорт г. Малмыж https://vk.com/public165382241": -165382241,
          "Аджимский Дом-Культуры https://vk.com/id420841463": 420841463,
          "МалмыЖ https://vk.com/club9363816": -9363816,
          "Игорь Степанов https://vk.com/stepanoigo": 225359471,
          "Социальное обслуживание в Малмыжском районе https://vk.com/kcsonm": -110599037,
          "Команда МотоБрат 43 enduro Малмыж": -206673654,
          "Балтач Волейбол Лигасы": -207949090,
          "СДК села Рожки Малмыжского района Кировской обла": -179595292,
          "Нурия Напольских": 49552683,
          "ДЮСШ Малмыж": -207481860,
          "Дом культуры с.Тат-Верх-Гоньба": -142669560,
          "Старотушкинская сельская библиотека имени Луки Гребнева": -145445694,
          "Место встречи - Гоньба": -955909,
          "Старотушкинский Сдк": 444820854,
          "СМЕШНОЕ ВИДЕО Самый позитивный паблик ВК https://vk.com/smexo": -132265,
          "Стань учёным! А ты возьми и стань! https://vk.com/becomeascientist": -197866857,
          "Время - вперёд! Только хорошие новости! https://vk.com/rossiya_segodnya": -65614662,
          "SciTopus Популяризируем популяризаторов https://vk.com/scitopus": -112289703,
          "НауЧпок Разные штуки по науке https://vk.com/nowchpok": -73083424,
          "The Batrachospermum Magazine журнальчик-водоросль https://vk.com/batrachospermum": -85330,
          "МУЗЫКА. МОТОР! Русские видеоклипы https://vk.com/russianmusicvideo": -37343149,
          "МУЗЫКА НУЛЕВЫХ СССР (СУПЕРДИСКОТЕКА 90х - 2000х) https://vk.com/public50638629": -50638629,
          "Музыка 70-х 80-х 90-х 2000-х.Саундтреки ! https://vk.com/public187135362": -187135362,
          "Лучшие Клипы": -143826701,
          "Культура & Искусство Журнал для умных и творческих https://vk.com/public31786047": -31786047,
          "Удивительный мир https://vk.com/ourmagicalworld": -42320333,
          "Случайный Ренессанс Искусство повсюду https://vk.com/accidental_renaissance": -92583139
          }

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

train = pd.read_csv('../../data/avoska_test_human.csv', header=None, names=['category', 'text'])
len_old_train = len(train.index)
list_train_old = train['text'].to_list()

count = 0
new_texts = []

for i in texts:
    count += 1
    if i not in list_train_old and i not in new_texts:
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

        train.loc[len(train.index)] = [a, i]
        new_texts.append(i)
    if count % 30 == 0:
        train.to_csv('../../data/temp.csv', header=False, encoding='utf-8', index=False)
    print(f"\033[3;30;42m    {summa}-{count}     \033[0;0m")

train = train.drop_duplicates('text', keep='last')
print("\033[3;30;42m  Было:\033[0;0m", len_old_train)
print("\033[3;30;42m  Стало:\033[0;0m", len(train.index))

train.to_csv('../../data/avoska_test_human.csv', header=False, encoding='utf-8', index=False)
