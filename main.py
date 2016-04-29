
import os, sys, os
import copy, pickle
from settings import *

import numpy as np

from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import LogisticRegression, SVM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from oridinal_clf import OrdClf
from utils import *
from utils import Timer

alpha_range = [0.1]

def load_ord(filename):
    with open(filename, 'r') as f:
        data = []
        for line in f.readlines():
            data.append([float(i) for i in line.split()])
        data = np.array(data)
    return data[:, :-1], data[:, -1]

def exp_once(qs, dataset, n_ords, quota, X_train, X_test, y_train, y_test):
    E_in_2, E_out_2 = [], []

    model = OrdClf(LogisticRegression(), n_ords)
    #model = OrdClf(SVM(C=0.1, kernel='linear', decision_function_shape='ovr'), n_ords)
    for i in range(quota) :
        print(qs, i)
        ask_id = qs.make_query()
        dataset.update(ask_id, y_train[ask_id])

        model.train(*(dataset.format_sklearn()))
        E_in_2 = np.append(E_in_2, model.score(*(dataset.format_sklearn())))
        E_out_2 = np.append(E_out_2, model.score(X_test, y_test))
    return E_in_2, E_out_2


def train(X_train, y_train, X_test, y_test, n_ords, start_quota, quota):
    ret = []

    model = OrdClf(LogisticRegression(), n_ords)
    model.train(X_train, y_train)
    print(model.score(X_test, y_test))

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

    #import ipdb; ipdb.set_trace()
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
                        ) for i in range(n_ords-1)],
                    #+ [
                    #KMeansRepresent(datasets[i], rep=i+1) for i in range(n_ords-1)
                    #],
                #models=[
                #     HintSVM(
                #         datasets[i],
                #         p=0.5,
                #         rep=i+1
                #         ) for i in range(n_ords-1)],
            )
        E_in, E_out = exp_once(qs, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
        ret.append(('randqs', E_in, E_out))

#=================================

    #for alpha in alpha_range:
    #    with Timer('linearUCB_test'+str(alpha)) as t:
    #        dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
    #        datasets = [Dataset(X_train, np.concatenate([y_train[:start_quota]<=(i+1), [None] * (len(y_train) - start_quota)])) for i in range(n_ords-1)]
    #        qs = LinUCBqs(
    #                dataset,
    #                clf=OrdClf(LogisticRegression(), n_ords),
    #                models=[
    #                    UncertaintySampling(
    #                        datasets[i],
    #                        model=LogisticRegression(),
    #                        method='lc',
    #                        rep=i+1
    #                        ) for i in range(n_ords-1)],\
    #                    #+ [
    #                    #KMeansRepresent(datasets[i], rep=i+1) for i in range(n_ords-1)
    #                    #],
    #                alpha=alpha,
    #                test_set = (X_test, y_test)
    #                )
    #        model = OrdClf(LogisticRegression(), n_ords)
    #        E_in, E_out = exp_once(qs, model, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
    #        ret.append(('linUCB_test'+str(alpha), E_in, E_out))

#=================================

    for alpha in alpha_range:
        with Timer('linearUCB'+str(alpha)) as t:
            dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
            datasets = [Dataset(X_train, np.concatenate([y_train[:start_quota]<=(i+1), [None] * (len(y_train) - start_quota)])) for i in range(n_ords-1)] + \
                    [Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))]
            print(len(datasets))
            #qs = LinUCBqs(
            #        dataset,
            #        clf=OrdClf(LogisticRegression(), n_ords),
            #        models=[
            #            UncertaintySampling(
            #                datasets[i],
            #                model=LogisticRegression(),
            #                method='lc',
            #                rep=i+1
            #                ) for i in range(n_ords-1)],\
            #            #+ [
            #            #KMeansRepresent(datasets[i], rep=i+1) for i in range(n_ords-1)
            #            #],
            #        alpha=alpha,
            #        )
            qs = Exp3(
                dataset,
                clf=OrdClf(LogisticRegression(), n_ords),
                 models=[
                     UncertaintySampling(
                         datasets[i],
                         model=LogisticRegression(),
                         method='lc',
                         rep=i+1
                         ) for i in range(n_ords-1)],
                        #+ [
                        #KMeansRepresent(datasets[i], rep=i+1) for i in range(n_ords-1)
                        #],
                gamma=alpha,
                quota=quota
            )
            #qs = BoundaryUncertain(
            ##qs = ModelChange(
            #    dataset,
            #    clf=OrdClf(LogisticRegression(), n_ords),
            #     models=[
            #         UncertaintySampling(
            #             datasets[i],
            #             model=LogisticRegression(),
            #             method='lc',
            #             rep=i+1
            #             ) for i in range(n_ords-1)],
            #     #        ) for i in range(n_ords-1)] + [
            #     #models=[
            #     #    HintSVM(
            #     #        datasets[i],
            #     #        p=0.5,
            #     #        rep=i+1
            #     #        ) for i in range(n_ords-1)],
            #     #        ) for i in range(n_ords-1)] + [
            #     #    KMeansRepresent(datasets[n_ords-1])
            #     #        ],
            #    quota=quota
            #)
            #qs = DWUS(dataset)
            E_in, E_out = exp_once(qs, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
            ret.append(('linUCB'+str(alpha), E_in, E_out))

#=================================

    with Timer('US') as t:
        dataset = Dataset(X_train, np.concatenate([y_train[:start_quota], [None] * (len(y_train) - start_quota)]))
        qs = UncertaintySampling(
                dataset,
                model=LogisticRegression(),
                #model=SVM(C=0.1, kernel='linear', decision_function_shape='ovr'),
                method='lc',
                )
        #qs = KMeansRepresent(dataset)

        E_in, E_out = exp_once(qs, dataset, n_ords, quota, X_train, X_test, y_train, y_test)
        ret.append(('us', E_in, E_out))

    return ret

def do_exp_once(dataset, n_ords, idx, gen_type):

    data_idx = data_name.index(dataset)
    name = data_name[data_idx]

    with open(out_paths[data_idx] % (gen_type, n_ords), 'rb') as f:
        X, y = pickle.load(f)
    X.astype('float')

    with open(BASE_PATH + '%s/%s_%s_bin%d.%d.pkl' % (name, name, gen_type, n_ords, idx), 'rb') as f:
    #with open(BASE_PATH + '%s/%s_eqw.%d.pkl' % (name, name, idx), 'rb') as f:
        split = pickle.load(f)

    start_quota = 20
    while len(np.unique(((y[split[0]])[:start_quota]))) != n_ords:
        np.random.shuffle(split[0])
    X_train, y_train = X[split[0]], y[split[0]]
    X_test, y_test = X[split[1]], y[split[1]]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #for i in range(1, n_ords+1):
    #    np.random.choice()

    print('X_train:', np.shape(X_train), 'X_test:', np.shape(X_test))

    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    #quota = len(split[0]) - start_quota
    quota = min(200, len(split[0]) - start_quota)

    return train(X_train, y_train, X_test, y_test, n_ords, start_quota, quota)

def mul_exps():
    #dataset = 'abalone'
    #dataset = 'stocksdomain'
    dataset = 'cpu_small'
    #dataset = 'bank8FM'
    #dataset = 'bank32nh'

    n_ords = 10
    #gen_type = 'eqw'
    gen_type = 'eqf'

    #for dataset in data_name:
    for dataset in ['cpu_small']:
    #for dataset in ['stocksdomain']:
    #for dataset in ['abalone']:
        for idx in range(10):
            result = do_exp_once(dataset, n_ords, idx, gen_type)

            result_path = '/tmp2/b01902066/ordinal_regression/results/' + dataset + '/'
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            print(result_path)
            #with open(result_path + '%s_bin%d_%d.pkl' % (dataset, n_ords, idx), 'wb') as f:
            with open(result_path + '%s_%s_bin%d_%d.pkl' % (dataset, gen_type, n_ords, idx), 'wb') as f:
                pickle.dump(result, f)


def main():
    if len(sys.argv) == 4:
        dataset = sys.argv[1]
        n_ords = int(sys.argv[2])
        idx = int(sys.argv[3])
    else:
        #dataset = 'abalone'
        dataset = 'stocksdomain'
        #dataset = 'cpu_small'
        n_ords = 10
        idx = 0
    mul_exps()
    exit()

    result = do_exp_once(dataset, n_ords, idx)

    result_path = '/tmp2/b01902066/ordinal_regression/results/' + dataset + '/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + '%s_bin%d_%d.pkl' % (dataset, n_ords, idx), 'wb') as f:
        pickle.dump(result, f)
    exit()

    #rets = Parallel(n_jobs=20, backend="threading")(delayed(do_exp_once)(i) for i in range(60))
    #results = {}
    #for ret in rets:
    #    for qs in ret:
    #        results.setdefault(qs[0], []).append(qs[2])


if __name__ == '__main__':
    #np.random.seed(1)
    main()
