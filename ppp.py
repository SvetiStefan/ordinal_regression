
import numpy as np
import pickle
import matplotlib.pyplot as plt


#with open('temp.pkl', 'rb') as f:
#    res = pickle.load(f)

with open('cpu_small_bin10.pkl', 'rb') as f:
    results = pickle.load(f)

#for i in results.keys():
#    results[i].extend(res[i])

query_num = np.arange(1, np.shape(results['us'])[1] + 1)
for res in results.keys():
    if res in ['us', 'randqs', 'linUCB100.0', 'linUCB_test100.0', 'random']:
        E_out = np.mean(results[res], axis=0)
        E_out_std = np.std(results[res], axis=0) / np.sqrt(60)
        #print(E_out_std)
        print(len(results[res]))
        #print(results[res])
        plt.plot(query_num, E_out, label=res)
        #plt.errorbar(query_num, E_out, yerr=E_out_std, label=res)

plt.xlabel('Number of Queries')
plt.ylabel('abs Error')
plt.title('< Experiment Result >')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
#plt.show()
plt.show(block=True)
