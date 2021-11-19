import re

import pymorphy2
from pytz import unicode

ma = pymorphy2.MorphAnalyzer()


def clean_text(text):
    d = str(text)
    d = re.sub("[^\w]", " ", d)
    d = re.sub("_", " ", d)
    d = d.lower()
    d = " ".join(ma.parse(unicode(word))[0].normal_form for word in d.split())
    d = re.sub(" анон", " ", d)
    d = re.sub("анон ", " ", d)
    d = re.sub("анонимно", " ", d)
    d = re.sub("аноним", " ", d)
    d = ' '.join(word for word in d.split() if len(word) > 3)
    d = list(set(d.split()))
    d = ' '.join(str(e) for e in d)
    if not d or d == '' or d == ' ' or d == '  ' or d in 'ананимноанонимно':
        d = ''

    return d
