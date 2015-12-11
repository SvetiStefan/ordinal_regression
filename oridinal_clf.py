
import numpy as np
from sklearn import tree
import pickle
from sklearn.cross_validation import train_test_split
from libact.models import LogisticRegression
#from sklearn.linear_model import LogisticRegression
from libact.base.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import pandas as pd
import copy
from utils import *

class OrdClf():
    def __init__(self, base_clf, n_ord):
        self.clfs = [copy.deepcopy(base_clf) for i in range(n_ord-1)]
        self.n_ord = n_ord

    def train(self, X, y):
        for i, clf in enumerate(self.clfs):
            clf.train(Dataset(X, (y<=(i+1))))
            #clf.fit(X, (y<=(i+1)))

    def predict(self, X):
        preds = []
        for i, clf in enumerate(self.clfs):
            preds.append(clf.predict(X))
        preds = np.array(preds).T
        return np.sum(1-preds, axis=1) + 1

        #ret = np.zeros((X.shape[0], self.n_ord))
        #for t in range(self.n_ord):
        #    ret[:, t] = np.sum(preds[:, :t] == 0, axis=1) +\
        #                np.sum(preds[:, t:] == 1, axis=1)

        #return np.argmax(ret, axis=1) + 1

    def score(self, X, y):
        return np.mean(np.abs(y - self.predict(X)))


def eval_score(y, yhat):
    return np.mean(np.abs(y - yhat))

def main():
    #with open('./data/housing.pkl', 'rb') as f:
    #    X, y = pickle.load(f)
    #X, y = load_ord('./data/cal_housing.data.ord')
    #split = train_test_split(range(X.shape[0]), test_size=0.5)

    #trn_X, trn_y, tst_X, tst_y = load_train_test('./data/stocksdomain/10bins/stock_train_10.1')
    with open('./data/cpu_small/cpu_small_bin10.pkl', 'rb') as f:
        X, y = pickle.load(f)
    print(y)
    #X, y = load_benchmark_data('./data/stocksdomain/10bins/stock_train_10.1')
    #X, y = load_benchmark_data('./data/Auto-Mpg/10bin/auto.data_train_10.1')
    #X, y = load_benchmark_data('./data/abalone/10bins/abalone_train_10.1')
    split = train_test_split(range(X.shape[0]), test_size=0.8)
    trn_X, trn_y, tst_X, tst_y = X[split[0]], y[split[0]], X[split[1]], y[split[1]]

    #scaler = MinMaxScaler()
    #X[split[0]] = scaler.fit_transform(X[split[0]])
    #X[split[1]] = scaler.transform(X[split[1]])

    #clf = OrdClf(tree.DecisionTreeClassifier(max_depth=4), 5)
    #clf = OrdClf(LinearSVC(), 5)
    clf = OrdClf(LogisticRegression(C=0.1), 10)
    clf.train(trn_X, trn_y)
    print(eval_score(trn_y, clf.predict(trn_X)))
    print(eval_score(tst_y, clf.predict(tst_X)))


if __name__ == '__main__':
    main()

