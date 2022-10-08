import pandas as pd
import seaborn as sn

train = pd.read_csv('data/titanic_train.csv', sep=',')
pd.set_option('display.max_columns', None)
print(train.head(10))
train.set_index('PassengerId', inplace=True, drop=True)
print(train.columns)