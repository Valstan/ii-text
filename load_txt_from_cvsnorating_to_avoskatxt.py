import pandas as pd


def main():
    in_txt = pd.read_csv('avoska_txt_norating.csv', header=None, names=['text'])
    out_txt = pd.read_csv('avoska_txt.csv', header=None, names=['category', 'text'])

    len_old_out = len(out_txt.index)
    list_in_txt = in_txt['text'].to_list()
    list_out_txt = out_txt['text'].to_list()
    summa = len(list_in_txt)

    count = 0
    new_texts = []
    for i in list_in_txt:
        count += 1
        if i not in list_out_txt and i not in new_texts:
            print(i)
            a = int(input(f"{summa}-{count} ... 0 or 1 or 666 - "))
            if a == 666:
                break
            out_txt.loc[len(out_txt.index)] = [a, i]
            new_texts.append(i)

    print("Было - ", len_old_out, " записей.")
    print("Стало - ", len(out_txt.index), " записей.")

    out_txt.to_csv('avoska_txt.csv', header=False, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
