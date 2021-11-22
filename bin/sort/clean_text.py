import re

import pymorphy2
from pytz import unicode

ma = pymorphy2.MorphAnalyzer()


def clean_text(text):
    d = str(text)
    d = d.lower()
    d = re.sub(r'\W_', " ", d)
    d = " ".join(ma.parse(unicode(word))[0].normal_form for word in d.split())
    d = re.sub(r"\bанон|анон\b|\bанан", " ", d)
    d = ' '.join(word for word in d.split() if len(word) > 3)
    d = list(set(d.split()))
    d = ' '.join(str(e) for e in d)

    return d
