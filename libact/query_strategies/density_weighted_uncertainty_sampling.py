"""Density Weighted Uncertainty Sampling
"""
import numpy as np
from sklearn.cluster import KMeans

from libact.base.interfaces import QueryStrategy, ContinuousModel
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


class DWUS(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(DWUS, self).__init__(*args, **kwargs)
        self.n_ordi = 10
        self.n_clusts = 10
        self.kmeans = KMeans(n_clusters=self.n_clusts)

        self.sigma = 0.5
        self.max_iter = 100

        all_x = np.array([xy[0] for xy in self.dataset.data])
        self.kmeans.fit(all_x)
        d = len(all_x[0])

        #x_k = self._p_x_k(all_x)

        #import ipdb; ipdb.set_trace()
        centers = self.kmeans.cluster_centers_
        P_k = np.ones(self.n_clusts) / float(self.n_clusts)

        dis = np.zeros((len(all_x), self.n_clusts))
        for i in range(self.n_clusts):
            dis[:, i] = np.exp(-np.einsum('ij,ji->i',
                                (all_x - centers[i]), (all_x - centers[i]).T) / 2 / self.sigma)
        for _ in range(self.max_iter):
            # E-step P(k|x)
            temp = dis * np.tile(P_k, (len(all_x), 1))
            # shape = (len(all_x), n_clusts)
            P_k_x = temp / np.tile(np.sum(temp, axis=1), (self.n_clusts, 1)).T

            # M-step
            P_k = 1./len(all_x) * np.sum(P_k_x, axis=0)

        self.P_k_x = P_k_x

        p_x_k = np.zeros((len(all_x), self.n_clusts))
        for i in range(self.n_clusts):
            p_x_k[:, i] = multivariate_normal.pdf(all_x, mean=centers[i], cov=np.ones(d)*np.sqrt(self.sigma))

        self.p_x = np.dot(p_x_k, P_k).reshape(-1)

    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        labeled_entry_ids = np.array([eid
            for eid, x in enumerate(self.dataset.data) if x[1] != None])
        labels = np.array([x[1]
            for eid, x in enumerate(self.dataset.data) if x[1] != None]).reshape(-1, 1)
        centers = self.kmeans.cluster_centers_
        P_k_x = self.P_k_x
        p_x = self.p_x[list(unlabeled_entry_ids)]

        #import ipdb; ipdb.set_trace()
        all_scores = np.zeros((self.n_ordi * len(p_x)))

        for i in range(self.n_ordi-1):
            clf = DensityWeightedLogisticRegression(P_k_x[labeled_entry_ids, :],
                                                    centers)
            clf.train(labeled_entry_ids, (labels <= (i+1)) * 2 - 1)
            P_y_k = clf.predict()

            P_y_x = np.zeros(len(unlabeled_entry_ids))
            for k, center in enumerate(centers):
                P_y_x += P_y_k[k] * P_k_x[unlabeled_entry_ids, k]

            # binary case
            expected_error = P_y_x
            expected_error[P_y_x >= 0.5] = 1. - P_y_x[P_y_x >= 0.5]

            all_scores[i*len(p_x): (i+1)*len(p_x)] = expected_error * p_x

            print(expected_error * p_x, np.max(expected_error * p_x))

        ask_id = np.argmax(all_scores)
        print(len(p_x), ask_id, int(ask_id / len(p_x)), ask_id % len(p_x))

        return unlabeled_entry_ids[ask_id%len(p_x)]

class DensityWeightedLogisticRegression(object):
    def __init__(self, density_estimate, centers, C=1.):
        self.density = density_estimate
        self.centers = centers
        self.C = C
        self.w = None

    def _likelihood(self, w, X, y):
        w = w.reshape(-1, 1)
        sigmoid = lambda t: 1. / (1. + np.exp(-t))
        # w --> shape = (d+1, 1)
        L = lambda w: (self.C/2. * np.dot(w[:-1].T, w[:-1]) - \
                np.sum(np.log(
                    np.sum(self.density *
                        sigmoid(np.dot(y, (np.dot(self.centers, w[:-1]) + w[-1]).T)
                        ), axis=1)
                ), axis=0))[0][0]

        return L(w)

    def train(self, X, y):
        d = np.shape(self.centers)[1]
        w = np.zeros((d+1, 1))
        sigmoid = lambda t: 1. / (1. + np.exp(-t))
        result = minimize(lambda _w: self._likelihood(_w, X, y),
                          w.reshape(-1),
                          method='CG',
                          options={'disp': True})
        w = result.x.reshape(-1, 1)
        self.w = w
        print(w)
        print(np.sum(np.log(
                np.sum(self.density *
                    sigmoid(np.dot(y, (np.dot(self.centers, w[:-1]) + w[-1]).T)
                    ), axis=1)
                ), axis=0))

    def predict(self):
        if self.w is not None:
            sigmoid = lambda t: 1. / (1. + np.exp(-t))
            return sigmoid(1. * np.dot(self.centers, self.w[:-1]) + self.w[-1])
