import pandas as pd

interval = 10
baza = pd.read_csv('../../data/avoska_lemms.csv', header=None, names=['category', 'text'])
all_text = baza['text']
all_data = len(all_text)
interval_bank = 0
print(all_data)

for i in range(0, 600, interval):
    count = 0
    b = i + interval
    for val in all_text:
        list_words = val.split()
        a = len(list_words)
        if i < a < b:
            count += 1
    interval_bank += count
    print(b, ' - ', count, ' - ', interval_bank, ' - ', all_data - interval_bank)
