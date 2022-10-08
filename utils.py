import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def parse_model_0(X):
    target = X.Survived
    X = X[['Fare', 'SibSp', 'Parch']]
    return X, target


def compute_score(clf, X, y):
    xval = cross_val_score(clf, X, y, cv=5)
    return np.mean(xval)


def plot_lr_coefs(X, lr: LogisticRegression):
    fig, ax = plt.subplots()

    xlabels = X.columns.values.tolist()
    yvalues = lr.coef_[0,]
    index = np.arange(len(yvalues))

    bar_width = 0.35
    opacity = 0.4
    rects = plt.bar(index, yvalues, bar_width, alpha=opacity, color='b', label='')
    plt.ylabel('Valeur')
    plt.title('Poids des variables')
    plt.xticks(index, xlabels, rotation=40)

    plt.legend()
    plt.tight_layout()
    plt.show()
