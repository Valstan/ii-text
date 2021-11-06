import json
import os


def number_words(txt_list):
    with open(os.path.join('number_words.json'), 'r', encoding='utf-8') as f:
        nw_list = json.load(f)

    box_numbers = []
    for i in txt_list:
        if i not in nw_list:
            nw_list.append(i)
        box_numbers.append(nw_list.index(i))

    with open(os.path.join('number_words.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(nw_list, indent=2, ensure_ascii=False))

    return box_numbers
