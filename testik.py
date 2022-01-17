import pandas as pd

# Укажите путь до папки
file = 'data/dubli.csv'

data = pd.read_csv(file, header=None, names=['category', 'text'])
new_data = pd.DataFrame(columns=['category', 'text'])

for sample in data['text']:
    try:
        if sample == new_data.iloc[-1:]['text'].tolist()[0]:
            continue
    except:
        pass
    dubl = data.loc[data['text'] == sample]
    if len(dubl.index) == 1:
        new_data = pd.concat([new_data, dubl], ignore_index=True)
    else:
        new_data = pd.concat([new_data, dubl[:1]], ignore_index=True)

new_data.to_csv('data/dubli2.csv', header=False, encoding='utf-8', index=False)


# print("\033[0;30;42m  Было - ", 23, " записей.    \033[0;0m")
# print('Цирки b кошки')
# print("\033[1;30;42m  Было - ", 23, " записей.    \033[0;0m")
# print('Цирки b кошки')
# print("\033[2;30;42m  Было - ", 23, " записей.    \033[0;0m")
# print('Цирки b кошки')
# print("\033[3;30;42m  Было - ", 23, " записей.    \033[0;0m")
# print('Цирки b кошки')
# print("\033[4;30;42m  Было - ", 23, " записей.    \033[0;0m")
# print('Цирки b кошки')
# print("\033[5;30;42m  Было - ", 23, " записей.    \033[0;0m")
# print('Цирки b кошки')
# summa = 22
# count = 333
# print(f"\033[3;30;42m  {summa} - {count}  \033[0;0m")
# print('\033[3;31mВведите 0 или 1, а вы ввели - \033[0;0m', 34)
# print('\033[3;31mКакая-то ошибка!!! Введи цифру 1 или 0\033[0;0m')
# print("\033[3;30;42m Было - ", 343, " записей. \033[0;0m")
# print("\033[3;30;42m Стало - ", 44332, " записей. \033[0;0m")
