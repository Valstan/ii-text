import easyocr
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from bin.rw.change_lp import change_lp
from bin.rw.get_image import image_get
from bin.rw.get_session import get_session
from bin.rw.get_session_vk_api import get_session_vk_api
from bin.rw.read_posts import read_posts
from bin.utils.clear_copy_history import clear_copy_history
from bin.utils.driver import load_table
from bin.utils.tesseract import tesseract

session = get_session('mi', 'config', 'novost')
session = load_table(session, session['name_session'])
session['groups'] = {"Подслушано Малмыж https://vk.com/podslyshanomalmyj": -149841761,
                     "Первый Малмыжский https://vk.com/malmiz": -86517261
                     }
vk_app = get_session_vk_api(change_lp(session))
posts = read_posts(vk_app, session['groups'], 10, 0)
new_posts = []
for i in posts:
    new_posts.append(clear_copy_history(i))

# ocr turnir
reader = easyocr.Reader(["ru", "en"])

for i in posts:
    height = 0
    url = ''
    if 'attachments' in i and 'photo' in i['attachments'][0]:
        for x in i['attachments'][0]['photo']['sizes']:
            if x['height'] > height:
                height = x['height']
                url = x['url']
        if image_get(url, '../image'):
            ocr = list(reader.readtext('image', detail=0, paragraph=False))
            tes = tesseract('image')
            if ocr:
                ocr = ' '.join(str(e) for e in ocr)
            print("ocr:\n", ocr, "\ntesseract:\n", tes)
            img = mpimg.imread('image')
            imgplot = plt.imshow(img)
            plt.show()
