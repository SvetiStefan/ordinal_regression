
import os, sys
import copy, pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from oridinal_clf import OrdClf
from utils import *
from joblib import Parallel, delayed

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
    #with open('./data/housing_bin10.pkl', 'rb') as f:
    #    X, y = pickle.load(f)
    with open('./data/cpu_small/cpu_small_bin10.pkl', 'rb') as f:
        X, y = pickle.load(f)

    # TODO problem??
    X = MinMaxScaler().fit_transform(X)

    #X, y = load_ord('./data/cal_housing.data.ord')
    #X, y = load_benchmark_data('./data/abalone/10bins/abalone_train_10.1')
    #X, y = load_benchmark_data('./data/stocksdomain/10bins/stock_train_10.1')
    #X, y = load_benchmark_data('./data/Auto-Mpg/10bin/auto.data_train_10.1')
    #X, y = load_benchmark_data('./data/bostonhousing/10bins/housing_train_10.1')

    split = train_test_split(range(X.shape[0]), test_size=0.8)
    #np.random.shuffle(split[0])
    #np.random.shuffle(split[1])

    X_train, y_train = X[split[0]], y[split[0]]
    #X_rew, y_rew = X[split[1][:int(len(split[1])/2)]], y[split[1][:int(len(split[1])/2)]]
    #X_test, y_test = X[split[1][int(len(split[1])/2):]], y[split[1][int(len(split[1])/2):]]
    X_test, y_test = X[split[1]], y[split[1]]

    n_ords = 10
    start_quota = 20
    #quota = N - start_quota
    quota = 400

    ret = []

    from utils import Timer

    with Timer('Rand') as t:
        E_in = []
        E_out = []
        model = OrdClf(LogisticRegression(), n_ords)
        for i in range(start_quota, start_quota+quota) :
            model.train(X_train[ : i + 1], y_train[ : i + 1])
            E_in.append(model.score(X_train[ : i + 1], y_train[ : i + 1]))
            E_out.append(model.score(X_test, y_test))
        ret.append(('random', E_in, E_out))

#=============================ALBL

    with Timer('RandQS') as t:
        dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
        datasets = [Dataset(X_train, np.concatenate([y_train[:start_quota]<=(i+1), [None] * (len(y_train) - start_quota)])) for i in range(n_ords-1)]
        qs = RandMulQs(
                dataset,
                models=[
                    UncertaintySampling(
                        datasets[i],
                        model=LogisticRegression(),
                        method='lc',
                        rep=i+1
                        ) for i in range(n_ords-1)
                    ],
            )
        model = OrdClf(LogisticRegression(), n_ords)
        E_in, E_out = exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
        ret.append(('randqs', E_in, E_out))

#=================================

    for alpha in [1000., 100., 10., 1., 0.1]:
        with Timer('linearUCB_test'+str(alpha)) as t:
            dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
            datasets = [Dataset(X_train, np.concatenate([y_train[:start_quota]<=(i+1), [None] * (len(y_train) - start_quota)])) for i in range(n_ords-1)]
            qs = LinUCB(
                    dataset,
                    clf=OrdClf(LogisticRegression(), n_ords),
                    models=[
                        UncertaintySampling(
                            datasets[i],
                            model=LogisticRegression(),
                            method='lc',
                            rep=i+1
                            ) for i in range(n_ords-1)
                        ],
                    alpha=alpha,
                    test_set = (X_test, y_test)
                    )
            model = OrdClf(LogisticRegression(), n_ords)
            E_in, E_out = exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
            ret.append(('linUCB_test'+str(alpha), E_in, E_out))

#=================================

    for alpha in [1000., 100., 10., 1., 0.1]:
        with Timer('linearUCB'+str(alpha)) as t:
            dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
            datasets = [Dataset(X_train, np.concatenate([y_train[:start_quota]<=(i+1), [None] * (len(y_train) - start_quota)])) for i in range(n_ords-1)]
            qs = LinUCB(
                    dataset,
                    clf=OrdClf(LogisticRegression(), n_ords),
                    models=[
                        UncertaintySampling(
                            datasets[i],
                            model=LogisticRegression(),
                            method='lc',
                            rep=i+1
                            ) for i in range(n_ords-1)
                        ],
                    alpha=alpha,
                    )
            model = OrdClf(LogisticRegression(), n_ords)
            E_in, E_out = exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
            ret.append(('linUCB'+str(alpha), E_in, E_out))
            if qs.test_set == None:
                print('Yes')

#=================================

    with Timer('US') as t:
        dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
        qs = UncertaintySampling(
                dataset,
                model=LogisticRegression(),
                method='lc',
                )
        model = OrdClf(LogisticRegression(), n_ords)
        E_in, E_out = exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
        ret.append(('us', E_in, E_out))

    return ret

def do_exp_once():
    while True:
        try:
            ret = train_and_plot()
            return ret
        except:
            continue

def main():
    results = {}
    rets = Parallel(n_jobs=20, backend="threading")(delayed(do_exp_once)() for i in range(60))
    #do_exp_once()

    for ret in rets:
        for qs in ret:
            results.setdefault(qs[0], []).append(qs[2])

    with open('cpu_small_bin10.pkl', 'wb') as f:
        pickle.dump(results, f)

    query_num = np.arange(1, 150 + 1)
    for res in results.keys():
        query_num = np.arange(1, 150 + 1)
        E_out = np.mean(results[res], axis=0)
        E_out_std = np.std(results[res], axis=0)
        #plt.plot(query_num, E_out, label=res)
        plt.errorbar(query_num, E_out, yerr=E_out_std, label=res)
    print('finished')

    plt.xlabel('Number of Queries')
    plt.ylabel('abs Error')
    plt.title('< Experiment Result >')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    #plt.show()
    plt.show(block=True)
    #plt.savefig('./png/' + str(i) + '.png')
    #plt.close()


if __name__ == '__main__':
    #np.random.seed(1)
    main()
