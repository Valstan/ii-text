import pandas as pd

# Укажите путь до папки
file = '../../data/avoska_human.csv'

data = pd.read_csv(file, header=None, names=['category', 'text'])
new_data = pd.DataFrame(columns=['category', 'text'])

for sample in data['text']:
    dubl = data.loc[data['text'] == sample]
    if len(dubl.index) == 1:
        new_data = pd.concat([new_data, dubl], ignore_index=True)
        new_data = new_data.sample(frac=1).reset_index(drop=True)
        continue
    print(dubl)
    # else:
    #     dubl_column = dubl.reset_index()
    #     dubl_column.columns = ['category', 'text']
    #     dubl = dubl_column.loc[dubl_column['category'] == dubl_column['category'][0]]
    #     dubl.columns = ['category', 'text']
    #     if dubl == dubl_column:
    #         new_data = pd.concat([new_data, dubl], ignore_index=True)
    #         new_data = new_data.sample(frac=1).reset_index(drop=True)
