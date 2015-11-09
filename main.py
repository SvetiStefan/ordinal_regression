
import os, sys
import copy, pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import LogisticRegression
from oridinal_clf import OrdClf

def load_ord(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f.readlines():
            data.append([float(i) for i in line.split()])
        data = np.array(data)
    return data[:, :-1], data[:, -1]


def exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test):
    E_in_2, E_out_2 = [], []

    model = OrdClf(LogisticRegression(), n_ords)
    for i in range(quota) :
        ask_id = qs.make_query()
        dataset.update(ask_id, y_train[ask_id])

        model.train(*(dataset.format_sklearn()))
        E_in_2 = np.append(E_in_2, model.score(*(dataset.format_sklearn())))
        E_out_2 = np.append(E_out_2, model.score(X_test, y_test))
    return E_in_2, E_out_2


def train_and_plot():
    #with open('./data/housing.pkl', 'rb') as f:
    #    X, y = pickle.load(f)
    with open('./data/housing_bin10.pkl', 'rb') as f:
        X, y = pickle.load(f)
    #X, y = load_ord('./data/cal_housing.data.ord')

    split = train_test_split(range(X.shape[0]), test_size=0.8)
    N = len(split[0])
    np.random.shuffle(split[0])
    np.random.shuffle(split[1])
    X_train, y_train = X[split[0]], y[split[0]]
    #X_rew, y_rew = X[split[1][:int(len(split[1])/2)]], y[split[1][:int(len(split[1])/2)]]
    #X_test, y_test = X[split[1][int(len(split[1])/2):]], y[split[1][int(len(split[1])/2):]]
    X_test, y_test = X[split[1]], y[split[1]]

    n_ords = 10
    start_quota = 15
    #quota = N - start_quota
    quota = 50

    trn_res = []
    tst_res = []

    #print('After randomly asking %d questions, (E_in, E_out) = (%f, %f)' % (N - start_quota, E_in_1[-1], E_out_1[-1]))
    for T in range(1):
        print(T)
        E_in_1, E_out_1 = [], []
        E_in_2, E_out_2 = [], []

        model = OrdClf(LogisticRegression(), n_ords)
        for i in range(start_quota, start_quota+quota) :
            model.train(X_train[ : i + 1], y_train[ : i + 1])
            E_in_1 = np.append(E_in_1, model.score(X_train[ : i + 1], y_train[ : i + 1]))
            E_out_1 = np.append(E_out_1, model.score(X_test, y_test))

#=============================ALBL
        for d in [2]:
            dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
            datasets = [Dataset(X_train, np.concatenate([y_train[:start_quota]<=(i+1), [None] * (len(y_train) - start_quota)])) for i in range(n_ords-1)]
            qs = ucb1(
                    dataset,
                    clf=OrdClf(LogisticRegression(), n_ords),
                    models=[
                        UncertaintySampling(
                            datasets[i],
                            method='lc',
                            rep=i+1
                            ) for i in range(n_ords-1)
                        ],
                    test_set = (X_test, y_test)
                    )
            #qs = ActiveLearningByLearning(
            #        dataset,
            #        #clf=LogisticRegression(),
            #        clf=OrdClf(LogisticRegression(), n_ords),
            #        delta=d,
            #        T=quota,
            #        models=[
            #            UncertaintySampling(
            #                datasets[i],
            #                method='lc',
            #                rep=i+1
            #                ) for i in range(n_ords-1)
            #            ],
            #        test_set = (X_test, y_test)
            #        )
            #qs.test_set = (X_rew, y_rew)
            #qs.test_set = (X_test, y_test)

            model = OrdClf(LogisticRegression(), n_ords)
            E_in, E_out = exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
            E_in_2.append(E_in)
            E_out_2.append(E_out)
            for i in range(quota) :
                print(E_in_1[i], E_out_1[i], E_in_2[-1][i], E_out_2[-1][i])

        trn_res.append([E_in_1] + E_in_2)
        print(np.shape(trn_res))
        tst_res.append([E_out_1] + E_out_2)
        for i in range(quota) :
            print(E_in_1[i], E_out_1[i], E_in_2[0][i], E_out_2[0][i])

    with open('./results/test_albl_bin5.pkl', 'wb') as f:
        pickle.dump((trn_res, tst_res), f)


    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='random_Ein')    # the blue curve
    plt.plot(query_num, E_in_2, 'r', label='albl_Ein')    # the red curve
    plt.plot(query_num, E_out_1, 'g', label='random_Eout')   # the green curve
    plt.plot(query_num, E_out_2, 'k', label='albl_Eout')   # the black curve
    plt.xlabel('Number of Queries')
    plt.ylabel('abs Error')
    plt.title('< Experiment Result >')
    plt.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0.)
    plt.show()
    #plt.show(block=False)
    #plt.savefig('./test.png')


def main():
    model_classname = None
    qs_classname = None
    model_params = {}
    qs_params = {}

    train_and_plot()


if __name__ == '__main__':
    #np.random.seed(1)
    main()
