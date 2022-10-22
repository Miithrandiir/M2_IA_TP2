import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import utils
import pandas as pd
import numpy as np


def plot_hist(feature, bins=20):
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label=['Victime', 'Survivant'], bins=bins)
    plt.legend(loc='upper left')
    plt.title('Distribution relative de %s' % feature)
    plt.show()


def parse_model_1(X):
    target = X.Survived
    class_dummies = pd.get_dummies(X['Pclass'], prefix='split_Pclass')
    X = X.join(class_dummies)
    to_del = ['Name', 'Age', 'Cabin', 'Embarked', 'Survived', 'Ticket', 'Sex', 'Pclass']
    for col in to_del:
        del X[col]
    return X, target


train = pd.read_csv('data/titanic_train.csv', sep=',')
train.set_index('PassengerId', inplace=True, drop=True)
survived = train[train.Survived == 1]
dead = train[train.Survived == 0]
plot_hist('Pclass')
lr = LogisticRegression(max_iter=1800)
X, y = parse_model_1(train.copy())
print("Moyenne des scores de la régression logistique : ", utils.compute_score(lr, X, y))
lr.fit(X, y)
utils.plot_lr_coefs(X, lr)
