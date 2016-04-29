
import numpy as np
import pickle
from sklearn.cross_validation import train_test_split

BASE_PATH = '/tmp2/b01902066/ordinal_regression/data/'

data_paths = [
    BASE_PATH + 'bank8FM/bank8FM.data',
    BASE_PATH + 'stocksdomain/stock',
    BASE_PATH + 'abalone/abalone',
    BASE_PATH + 'bostonhousing/housing',
    BASE_PATH + 'bank32nh/bank32nh.data',
    BASE_PATH + 'cpu_small/cpu_small.data',
    BASE_PATH + 'cal_housing/cal_housing.data',
    BASE_PATH + 'house_8L/house_8L.data',
]

out_paths = [
    BASE_PATH + 'bank8FM/bank8FM_%s_bin%d.pkl',
    BASE_PATH + 'stocksdomain/stocksdomain_%s_bin%d.pkl',
    BASE_PATH + 'abalone/abalone_%s_bin%d.pkl',
    BASE_PATH + 'bostonhousing/bostonhousing_%s_bin%d.pkl',
    BASE_PATH + 'bank32nh/bank32nh_%s_bin%d.pkl',
    BASE_PATH + 'cpu_small/cpu_small_%s_bin%d.pkl',
    BASE_PATH + 'cal_housing/cal_housing_%s_bin%d.pkl',
    BASE_PATH + 'house_8L/house_8L_%s_bin%d.pkl',
]

data_name = [
    'bank8FM',
    'stocksdomain',
    'abalone',
    'bostonhousing',
    'bank32nh',
    'cpu_small',
    'cal_housing',
    'house_8L',
]

def gen_split():
    import itertools
    for ty, n_bins in itertools.product(['eqf', 'eqw'], [5, 10]):
        for idx, path in enumerate(out_paths):
            name = data_name[idx]
            print(name)
            with open(path % (ty, n_bins,), 'rb') as f:
                X, y = pickle.load(f)
            print(np.sum(y==1), np.sum(y==5), np.sum(y==10))
            n_ord = len(np.unique(y))
            for i in range(60):
                split = train_test_split(range(X.shape[0]), test_size=0.5)
                while len(np.unique(y[split[0]])) != n_ord:
                    split = train_test_split(range(X.shape[0]), test_size=0.5)

                with open(BASE_PATH + '%s/%s_%s_bin%d.%d.pkl' % (name, name, ty, n_bins, i), 'wb') as f:
                    pickle.dump(split, f)
    print('done')

def eqfreq():
    for n_bins in [5, 10]:
        for idx, path in enumerate(data_paths):
            name = data_name[idx]
            print(name)
            with open(path, 'r') as f:
                data = []
                for k, line in enumerate(f.readlines()):
                    if line == '\n': break
                    data.append([float(i.strip()) for i in line.split(',')])
                data = np.array(sorted(data, key=lambda x: x[-1]))
                print(np.shape(data))
                for i in range(n_bins):
                    data[int(i*float(len(data))/n_bins): int((i+1)*float(len(data))/n_bins), -1] = i+1

                with open(BASE_PATH + '%s/%s_eqf_bin%d.pkl' % (name, name, n_bins), 'wb') as fo:
                    pickle.dump((data[:, :-1], data[:, -1]), fo)

def eqwidth():
    for n_bins in [5, 10]:
        for idx, path in enumerate(data_paths):
            name = data_name[idx]
            print(name)
            with open(path, 'r') as f:
                data = []
                lbl = []
                for k, line in enumerate(f.readlines()):
                    if line == '\n': break
                    data.append([float(i.strip()) for i in line.split(',')])
                    lbl.append(float(line.split(',')[-1]))
                l = min(lbl)
                r = max(lbl) + 0.0001
                step = (r - l) / n_bins

                data = np.array(data)
                print(np.shape(data))
                for i in range(n_bins):
                    s = np.logical_and((data[:, -1] >= (l + i * step)), (data[:, -1] < (l + (i+1) * step)))
                    data[s, -1] = i+1

                with open(BASE_PATH + '%s/%s_eqw_bin%d.pkl' % (name, name, n_bins), 'wb') as fo:
                    pickle.dump((data[:, :-1], data[:, -1]), fo)

def main():
    #eqfreq()
    #eqwidth()
    gen_split()
    exit()


if __name__ == '__main__':
    main()
