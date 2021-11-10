import easyocr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from bin.rw.get_image import image_get
from bin.utils.tesseract import tesseract
from get_posts_with_image import get_posts_with_image

posts = get_posts_with_image(100, 20)

# ocr turnir
reader = easyocr.Reader(["ru", "en"])

for i in posts:

    for x in i['attachments'][0]['photo']['sizes']:
        height = 0
        url = ''
        if x['height'] > height:
            height = x['height']
            url = x['url']
    if image_get(url, 'image'):
        ocr = list(reader.readtext('image', detail=0, paragraph=False))
        tes = tesseract('image')
        if ocr:
            ocr = ' '.join(str(e) for e in ocr)
        print("ocr:\n", ocr, "\ntesseract:\n", tes)
        img = mpimg.imread('image')
        imgplot = plt.imshow(img)
        plt.show()

