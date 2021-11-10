import easyocr

reader = easyocr.Reader(["ru", "en"])


def free_ocr(path_image):
    text = list(reader.readtext(path_image, detail=0))
    if text:
        return ' '.join(str(e) for e in text)
