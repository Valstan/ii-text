import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'c:\\tesseract5\\tesseract.exe'


def tesseract(patch):
    img = cv2.imread(patch)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ret, threshold_image = cv2.threshold(img, 127, 255, 0)
    custom_config = r'--oem 3 --psm 6'
    try:
        return pytesseract.image_to_string(threshold_image, lang='rus', config=custom_config)
    except:
        return ''
