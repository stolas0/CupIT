import pandas as pd
import ranging
import classification

from preparing import prepare

# ==============
# CLASSIFICATION
# ==============

train = pd.read_csv('../CupIT2019/Classification/train_data.csv', encoding='utf', engine='python', index_col=0)
test = pd.read_csv('../CupIT2019/Classification/test_data.csv', encoding='utf', engine='python', index_col=0)

train = prepare(train, 'Classification', fltr=r'<.*?>|[^а-яА-Яa-zA-Z ёЁ]')
test = prepare(test, 'Classification', fltr=r'<.*?>|[^а-яА-Яa-zA-Z ёЁ]')
print('2')
classification.run(train, test)

# =======
# RANGING
# =======

train = pd.read_csv('../CupIT2019/Ranging/train_data.csv', encoding='utf', engine='python', index_col=0)
test = pd.read_csv('../CupIT2019/Ranging/test_data.csv', encoding='utf', engine='python', index_col=0)

train = prepare(train, 'Ranging', fltr=r'<.*?>|[^а-яА-Яa-zA-Z ёЁ]', morph=False)
test = prepare(test, 'Ranging', fltr=r'<.*?>|[^а-яА-Яa-zA-Z ёЁ]', morph=False)

ranging.run(train, test)
