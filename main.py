import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import utils


train = pd.read_csv('data/titanic_train.csv', sep=',')
pd.set_option('display.max_columns', None)
print(train.head(10))
train.set_index('PassengerId', inplace=True, drop=True)
print(train.columns)

X, y = utils.parse_model_0(train.copy())
print("Moyenne des scores de la régression logistique : ", utils.compute_score(LogisticRegression(), X,y))