
import numpy as np
from sklearn.cluster import KMeans

from libact.base.interfaces import QueryStrategy


class KMeansRepresent(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(KMeansRepresent, self).__init__(*args, **kwargs)

        self.clusters = kwargs.pop('clusters', 5)
        self.kmeans = KMeans(n_clusters=self.clusters)
        self.rep = kwargs.pop('rep', None)

    def make_query(self):
        dataset = self.dataset

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        #self.kmeans.fit(X_pool)
        #distances = [abs(self.kmeans.score(x.reshape(1, -1))) for x in X_pool]
        #self.scores = [x for x in zip(
        #    unlabeled_entry_ids, distances
        #)]
        #ask_id = np.argmin(distances)

        #return unlabeled_entry_ids[ask_id]

        distances = self.kmeans.fit_transform(X_pool)
        self.scores = [x for x in zip(
            unlabeled_entry_ids, np.min(distances, axis=1)
        )]
        ask_id = np.argmin(np.min(distances, axis=1))

        return unlabeled_entry_ids[ask_id]

    def get_score(self, X):
        dist = self.kmeans.transform(X)
        return np.min(dist, axis=1)

