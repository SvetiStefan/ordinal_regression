
import numpy as np
import pickle

def main():
    for n_bins in [5, 10]:
        #with open('./housing.data', 'r') as f:
        #with open('./house_8L/house_8L.data', 'r') as f:
        #with open('./cpu_small/cpu_small.data', 'r') as f:
        #with open('./cal_housing/cal_housing.data', 'r') as f:
        with open('./bank8FM/bank8FM.data', 'r') as f:
            data = []
            for line in f.readlines():
                #data.append([float(i.strip()) for i in line.split(',')])
                data.append([float(i.strip()) for i in line.split()])
            data = np.array(sorted(data, key=lambda x: x[-1]))
            print(np.shape(data))
            for i in range(n_bins):
                data[int(i*float(len(data))/n_bins): int((i+1)*float(len(data))/n_bins), -1] = i+1

            #with open('./housing_bin5.pkl', 'wb') as fo:
            #with open('./house_8L/house_8L_bin%d.pkl' % n_bins, 'wb') as fo:
            #with open('./cpu_small/cpu_small_bin%d.pkl' % n_bins, 'wb') as fo:
            #with open('./cal_housing/cal_housing_bin%d.pkl' % n_bins, 'wb') as fo:
            with open('./bank8FM/bank8FM_bin%d.pkl' % n_bins, 'wb') as fo:
                pickle.dump((data[:, :-1], data[:, -1]), fo)


if __name__ == '__main__':
    main()
