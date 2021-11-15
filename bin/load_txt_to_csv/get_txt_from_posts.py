import re

import pymorphy2
from pytz import unicode

from bin.rw.get_image import image_get
from bin.utils.free_ocr import free_ocr

ma = pymorphy2.MorphAnalyzer()


def get_txt_from_posts(posts):
    texts = []
    count_pictures = 0
    for i in posts:
        if 'text' in i and i['text']:
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

    return texts
