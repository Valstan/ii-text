import pandas as pd

# Укажите путь до папки
test_file = '../../data/avoska_test_lemms.csv'
human_file = '../../data/avoska_lemms.csv'

test = pd.read_csv(test_file, header=None, names=['category', 'text'])
old_count_test = len(test.index)
list_category = test['category'].to_list()
list_text_test = test['text'].to_list()

human = pd.read_csv(human_file, header=None, names=['category', 'text'])
old_count_human = len(human.index)

for idx, val in enumerate(list_text_test):
    if idx % 100 == 0:
        print(idx)
    string_text = str(val)
    if string_text:
        human.loc[len(human.index)] = [list_category[idx], string_text]

test = test.iloc[0:0]
# human = human.drop_duplicates('text', keep='last')

print('Было данных - ', old_count_human)
print('Новых данных - ', old_count_test)
print('Стало данных - ', len(human.index))

# Сохраняем все в файлы
human.to_csv(human_file, header=False, encoding='utf-8', index=False)
test.to_csv(test_file, header=False, encoding='utf-8', index=False)
