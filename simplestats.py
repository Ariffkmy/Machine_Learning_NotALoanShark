import pandas as pd

data = pd.read_csv('encoded_dataset.csv',encoding='utf-8')
statistics = data.describe(include='all')

print(statistics)