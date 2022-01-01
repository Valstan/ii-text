import pandas as pd

baza = pd.read_csv('data/avoska_udpipe_dict.csv', header=None, names=['category', 'text'])
all_text = baza['text']
print(len(all_text))
for i in range(0, 600, 50):
    count = 0
    for val in all_text:
        list_words = val.split()
        a = len(list_words)
        b = i + 50
        if i < a < b:
            count += 1
    print(i, ' - ', count)
