#!/usr/bin/env python3
# coding: utf-8

import os
import re
import sys

import pandas as pd
import pymorphy2
import wget
from pytz import unicode
from ufal.udpipe import Model, Pipeline
from natasha import MorphVocab
from natasha import Doc

"""
Этот скрипт принимает на вход необработанный русский текст 
(одно предложение на строку или один абзац на строку).
Он токенизируется, лемматизируется и размечается по частям речи с использованием UDPipe.
На выход подаётся последовательность разделенных пробелами лемм с частями речи 
("зеленый_ADJ трамвай_NOUN").
Их можно непосредственно использовать в моделях с RusVectōrēs (https://rusvectores.org).
Примеры запуска:
echo 'Мама мыла раму.' | python3 rus_preprocessing_udpipe.py
zcat large_corpus.txt.gz | python3 rus_preprocessing_udpipe.py | gzip > processed_corpus.txt.gz
"""


def num_replace(word):
    # newtoken = "x" * len(word)
    newtoken = "цифра"
    return newtoken


def list_replace(search, replacement, text):
    search = [el for el in search if el in text]
    for c in search:
        text = text.replace(c, replacement)
    return text


def unify_sym(text):  # принимает строку в юникоде
    text = list_replace(
        "\u00AB\u00BB\u2039\u203A\u201E\u201A\u201C\u201F\u2018\u201B\u201D\u2019",
        "\u0022",
        text,
    )

    text = list_replace(
        "\u2012\u2013\u2014\u2015\u203E\u0305\u00AF", "\u2003\u002D\u002D\u2003", text
    )

    text = list_replace("\u2010\u2011", "\u002D", text)

    text = list_replace(
        "\u2000\u2001\u2002\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u202F\u205F\u2060\u3000",
        "\u2002",
        text,
    )

    text = re.sub("\u2003\u2003", "\u2003", text)
    text = re.sub("\t\t", "\t", text)

    text = list_replace(
        "\u02CC\u0307\u0323\u2022\u2023\u2043\u204C\u204D\u2219\u25E6\u00B7\u00D7\u22C5\u2219\u2062",
        ".",
        text,
    )

    text = list_replace("\u2217", "\u002A", text)

    text = list_replace("…", "...", text)

    text = list_replace("\u2241\u224B\u2E2F\u0483", "\u223D", text)

    text = list_replace("\u00C4", "A", text)  # латинская
    text = list_replace("\u00E4", "a", text)
    text = list_replace("\u00CB", "E", text)
    text = list_replace("\u00EB", "e", text)
    text = list_replace("\u1E26", "H", text)
    text = list_replace("\u1E27", "h", text)
    text = list_replace("\u00CF", "I", text)
    text = list_replace("\u00EF", "i", text)
    text = list_replace("\u00D6", "O", text)
    text = list_replace("\u00F6", "o", text)
    text = list_replace("\u00DC", "U", text)
    text = list_replace("\u00FC", "u", text)
    text = list_replace("\u0178", "Y", text)
    text = list_replace("\u00FF", "y", text)
    text = list_replace("\u00DF", "s", text)
    text = list_replace("\u1E9E", "S", text)

    currencies = list(
        "\u20BD\u0024\u00A3\u20A4\u20AC\u20AA\u2133\u20BE\u00A2\u058F\u0BF9\u20BC\u20A1\u20A0\u20B4\u20A7\u20B0\u20BF\u20A3\u060B\u0E3F\u20A9\u20B4\u20B2\u0192\u20AB\u00A5\u20AD\u20A1\u20BA\u20A6\u20B1\uFDFC\u17DB\u20B9\u20A8\u20B5\u09F3\u20B8\u20AE\u0192"
    )

    alphabet = list(
        '\t\n\r абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ,.[]{}()=+-−*&^%$#@!?~;:0123456789§/\|"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
    )

    alphabet.append("'")

    allowed = set(currencies + alphabet)

    cleaned_text = [sym for sym in text if sym in allowed]
    cleaned_text = "".join(cleaned_text)

    return cleaned_text


morph_vocab = MorphVocab()


# Укажите путь до файлов
data_path = '../../data/'
texts = 'avoska_test_human.csv'
lemms = 'avoska_test_morf.csv'

data = pd.read_csv(data_path + texts, header=None, names=['category', 'text'])
old_count_human = len(data.index)
list_category = data['category'].to_list()
list_text_human = data['text'].to_list()

data = pd.read_csv(data_path + lemms, header=None, names=['category', 'text'])
old_count_udpipe = len(data.index)
data = data.iloc[0:0]

for idx, val in enumerate(list_text_human):
    if idx % 100 == 0:
        print(idx)
    string_text = str(val)
    # Чистка текста от мусора
    res = unify_sym(string_text.strip())
    doc = Doc(res)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    output = list(ma.parse(unicode(word))[0].normal_form for word in res.split())
    output = ' '.join(word for word in output if len(word) > 2)
    if output:
        data.loc[len(data.index)] = [list_category[idx], output]

# data = data.drop_duplicates('text', keep='last')
data = data[data['text'].str.strip().astype(bool)]

print('\033[3;30;42m Данных сырых:\033[0;0m', old_count_human)
print('\033[3;30;42m Было данных LEMMS:\033[0;0m', old_count_udpipe)
print('\033[3;30;42m Стало данных LEMMS:\033[0;0m', len(data.index))

# Сохраняем все в файлы
data.to_csv(data_path + lemms, header=False, encoding='utf-8', index=False)
