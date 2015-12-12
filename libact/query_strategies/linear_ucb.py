

from libact.base.interfaces import QueryStrategy
import numpy as np
import copy, math

class LinUCB(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(LinUCB, self).__init__(*args, **kwargs)

        self.models_ = kwargs.pop('models', None)
        if self.models_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )
        elif not self.models_:
            raise ValueError("models list is empty")

        self.clf = kwargs.pop('clf', None)
        if self.clf is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )

        self.test_set = kwargs.pop('test_set', None)

        # n_actions <-- n_unlabeled_data

        # n_features <-- n_models
        self.d = len(self.models_) + 1

        self.alpha = kwargs.pop('alpha', 0.1)

        self.linucb_ = None

        self.hist_id = []
        self.hist_score = []
        self.hist_lbl = []

    def update(self, entry_id, label):
        self.hist_lbl.append(label)
        self.reward = self.calc_reward(entry_id, label)
        for model in self.models_:
            model.dataset.update(entry_id, (label<=model.rep))

    def calc_reward(self, entry_id, label):
        #clf = copy.deepcopy(self.clf)
        clf = self.clf
        clf.train(*(self.dataset.format_sklearn()))

        if self.test_set == None:
            costs = np.abs(np.array(self.hist_lbl) - clf.predict(np.array([self.dataset.data[i[1]][0] for i in self.hist])))
            return -1 * costs * np.array(self.hist_score)

            #reward = 0.
            #costs = np.abs(np.array([i[2] for i in self.hist]) -
            #               clf.predict(np.array([self.dataset.data[i[1]][0] for i in self.hist])))
            #for i, s in enumerate(self.hist):
            #    reward += -1./s[0] * costs[i]
            #return reward
        else:
            cost = clf.score(self.test_set[0], self.test_set[1])
            return -cost

    def linucb(self, x):
        A = np.eye(self.d)
        b = np.ones(self.d) / self.d

        while True:
            invA = np.linalg.pinv(A)
            theta = np.dot(invA, b)

            n_fet = x.shape[1]
            p = np.zeros(n_fet)
            for a in range(n_fet):
                p[a] = np.dot(theta, x[:, a]) + self.alpha * \
                    np.sqrt(np.dot(np.dot(x[:, a].T, invA), x[:, a]))

            #p = np.dot(theta, x) + \
            #    self.alpha * np.sqrt(np.einsum('ij,ji->i', np.dot(x.T, invA), x))

            # next feature x and last reward
            at = np.argmax(p)
            x_new, r = yield at, p[at]

            A = A + np.dot(x[:, at], x[:, at].T)
            b = b + r * x[:, at]

            x = x_new

    def make_query(self):
        dataset = self.dataset
        entry_id_to_idx = {}
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        for i, ids in enumerate(unlabeled_entry_ids):
            entry_id_to_idx[ids] = i

        queries = []
        X = []
        for model in self.models_:
            queries.append(model.make_query())
            assert len(model.scores) == len(entry_id_to_idx)
            score = [0 for i in range(len(entry_id_to_idx))]
            for s in model.scores:
                score[entry_id_to_idx[s[0]]] = s[1]
            X.append(score)
        X.append([1. for i in range(len(entry_id_to_idx))])

        if self.linucb_ == None:
            self.linucb_ = self.linucb(np.array(X))
            ask_idx, s = next(self.linucb_)
        else:
            ask_idx, s = self.linucb_.send((np.array(X), self.reward))

        #self.hist.append([s, unlabeled_entry_ids[ask_idx], -1])
        self.hist_id.append(unlabeled_entry_ids[ask_idx])
        self.hist_score.append(s)

        return unlabeled_entry_ids[ask_idx]


