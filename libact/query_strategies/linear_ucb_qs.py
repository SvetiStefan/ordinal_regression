
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.base.interfaces import QueryStrategy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import copy, math

class LinUCBqs(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(LinUCBqs, self).__init__(*args, **kwargs)

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

        # n_actions <-- models

        # n_features <-- unlabeled_data
        self.d = len(self.dataset) + 1

        self.alpha = kwargs.pop('alpha', 0.1)

        self.linucb_ = None

        self.hist_id = []
        self.hist_score = []
        self.hist_lbl = []

        self.reward = 1.
        self.cost = 1.

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
            #costs = np.abs(np.array(self.hist_lbl) -
            #               clf.predict(np.array([self.dataset.data[i][0] for i in self.hist_id])))
            ##return np.sum(costs / np.array(self.hist_score))
            ##return (10.-np.mean(costs)) / 10.
            #ret = self.cost - (10.-np.mean(costs)) / 10.
            #self.cost = (10.-np.mean(costs)) / 10.

            clf = LogisticRegression()
            dataset = self.dataset
            dataset.data[entry_id] = (dataset.data[entry_id][0], None)
            X, y = zip(*dataset.get_labeled_entries())
            clf.train(Dataset(X, y<=self.ask_qs))
            ori_prob = clf.predict_real(
                np.array([self.dataset.data[i][0] for i in self.hist_id]))

            dataset.data[entry_id] = (dataset.data[entry_id][0], label)
            X, y = zip(*dataset.get_labeled_entries())
            clf.train(Dataset(X, y<=self.ask_qs))
            prob = clf.predict_real(
                np.array([self.dataset.data[i][0] for i in self.hist_id]))

            print(ori_prob)
            #print(prob)
            ret = np.mean(np.abs(ori_prob-prob))
        else:
            cost = clf.score(self.test_set[0], self.test_set[1])
            #return (10.-cost) / 10.
            ret =  self.cost - (10.-cost) / 10.
            self.cost = (10.-cost) / 10.

        print(ret, self.cost)
        return ret

    def linucb(self, x):
#choose points
        A = np.eye(self.d)
        b = np.ones(self.d) / np.sqrt(float(self.d))

        while True:
            invA = np.linalg.pinv(A)
            theta = np.dot(invA, b)

            p = np.dot(x, theta) + \
                self.alpha * np.sqrt(np.einsum('ij,ji->i', np.dot(x, invA), x.T))

            # next feature x and last reward
            at = np.random.choice(np.where(p == np.max(p))[0])
            #at2 = np.argmax(p)
            x_new, r = yield at, p[at]
            print(r, p)
            #print(theta, np.dot(x, theta), p, r)

            A = A + np.dot(x[at, :], x[at, :].T)
            b = b + r * x[at, :]

            x = x_new

    #def linucb(self, x):

    def make_query(self):
        dataset = self.dataset
        entry_id_to_idx = {}
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        all_entry = np.array([entry[0] for entry in dataset.data])
        print(all_entry.shape)

        queries = []
        X = []
        for model in self.models_:
            queries.append(model.make_query())
            score = model.get_score(all_entry)
            score = np.append(score, 1.)
            X.append(score)
        X = np.array(X)
        self.scores = X
        # shape (fet, instances)
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        X[:, :-1] = scaler.fit_transform(X[:, :-1])
        #X[:-1, :] = scaler.fit_transform((X.T)[:, :-1]).T
        #for i in range(X.shape[0]):
        #    X[i, :] /= np.linalg.norm(X[i, :])

        if self.linucb_ == None:
            self.linucb_ = self.linucb(np.array(X))
            ask_qs, s = next(self.linucb_)
        else:
            ask_qs, s = self.linucb_.send((np.array(X), self.reward))

        print('ask:', ask_qs)
        self.ask_qs = ask_qs
        ask_idx = queries[ask_qs]
        self.hist_id.append(ask_idx)
        self.hist_score.append(s)

        return ask_idx


