import utils
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def plot_hist(feature, bins=20):
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label=['Victime', 'Survivant'], bins=bins)
    plt.legend(loc='upper left')
    plt.title('Distribution relative de %s' % feature)
    plt.show()


def fix_missing_value(data):
    # median for female
    # female = data[data['Sex'] == 'female']
    # female['Age'] = female['Age'].fillna(np.nanmedian(female['Age']))
    # male = data[data['Sex'] == 'male']
    # male['Age'] = male['Age'].fillna(np.nanmedian(male['Age']))
    # print(female, male)
    # return pd.concat([female, male])
    data.Age = data.Age.fillna(np.nanmedian(data["Age"]))
    return data


def split_value(data):
    target = data.Survived
    del data['Survived']
    X = data
    class_dummies = pd.get_dummies(X['Sex'], prefix='split_Sexe')
    X = X.join(class_dummies)
    class_dummies = pd.get_dummies(X['Pclass'], prefix='split_Pclass')
    X = X.join(class_dummies)
    X['is_child'] = X.Age < 8
    to_del = ['Name', 'Cabin', 'Embarked', 'Ticket', 'Sex', 'Pclass']
    for col in to_del:
        del X[col]
    return X, target


train = pd.read_csv('data/titanic_train.csv', sep=',')
train.set_index('PassengerId', inplace=True, drop=True)
train = fix_missing_value(train)
survived = train[train.Survived == 1]
dead = train[train.Survived == 0]
plot_hist('Age')
lr = LogisticRegression(max_iter=2000)
X, y = split_value(train.copy())
lr.fit(X, y)
print(utils.compute_score(lr, X,y))
utils.plot_lr_coefs(X, lr)