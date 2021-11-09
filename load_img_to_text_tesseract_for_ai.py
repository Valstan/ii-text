import pandas as pd

from bin.rw.get_image import image_get
from bin.utils.tesseract import tesseract
from get_posts_with_image import get_posts_with_image

posts = get_posts_with_image(200, 100)

texts = []
count_pictures = 0
for i in posts:

    for x in i['attachments'][0]['photo']['sizes']:
        height = 0
        url = ''
        if x['height'] > height:
            height = x['height']
            url = x['url']
    if image_get(url, 'image'):
        a = tesseract('image')
        count_pictures += 1
        if count_pictures % 10:
            print('*', end='')
        else:
            print('\nКартинок распознано - ', count_pictures)
        if a:
            texts.append(a)
print('\nИтого картинок распознано - ', count_pictures)
train = pd.read_csv('avoska_tesseract_txt.csv', header=None, names=['category', 'text'])
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

train.to_csv('avoska_tesseract_txt.csv', header=False, encoding='utf-8', index=False)
