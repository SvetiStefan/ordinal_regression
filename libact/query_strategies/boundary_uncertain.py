
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy, math

class BoundaryUncertain(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(BoundaryUncertain, self).__init__(*args, **kwargs)
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
                "__init__() missing required keyword-only argument: 'clf'"
                )

        self.reward = -1
        self.cost = 1.
        self.w = [1. for i in range(len(self.models_))]
        self.n_ord = 10

        self.hist_id = []
        self.hist_lbl = []
        self.hist_proba = []
        self.rewards = []

    def boundary_diff(self, entry_id, label, boundary_num):

        clf = LogisticRegression()
        dataset = self.dataset
        dataset.data[entry_id] = (dataset.data[entry_id][0], None)
        X, y = zip(*dataset.get_labeled_entries())
        clf.train(Dataset(X, y<=(boundary_num+1)))
        #print(y<=(boundary_num+1))
        #clf.train(Dataset(X, y<=((self.ask_qs%9)+1)))
        #ori_prob = clf.predict_real(
        #    np.array([self.dataset.data[i][0] for i in self.hist_id]))[:, 0]
        ori_prob = clf.predict_real(
            np.array([self.dataset.data[i][0] for i in range(len(self.dataset.data))]))[:, 0]

        dataset.data[entry_id] = (dataset.data[entry_id][0], label)
        X, y = zip(*dataset.get_labeled_entries())
        clf.train(Dataset(X, y<=(boundary_num+1)))
        #clf.train(Dataset(X, y<=((self.ask_qs%9)+1)))
        #prob = clf.predict_real(
        #    np.array([self.dataset.data[i][0] for i in self.hist_id]))[:, 0]
        prob = clf.predict_real(
            np.array([self.dataset.data[i][0] for i in range(len(self.dataset.data))]))[:, 0]

        ret = np.mean(np.abs(ori_prob-prob))
        return ret

    def calc_reward(self, entry_id, label):
        #import ipdb; ipdb.set_trace()
        #if self.ask_qs == self.n_ord-1:
        #    difs = [self.boundary_diff(entry_id, label, np.int64(i)) for i in range(self.n_ord-1)]
        #    ret = np.min(difs)
        #else:
        #    ret = self.boundary_diff(entry_id, label, self.ask_qs)
        ret = self.boundary_diff(entry_id, label, self.ask_qs)

        return ret

    def update(self, entry_id, label):
        self.hist_id.append(entry_id)
        self.hist_lbl.append(label)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        reward = self.calc_reward(entry_id, label)
        self.w[self.ask_qs] = reward
        #self.reward = sigmoid(self.calc_reward(entry_id, label) - 0.5)
        for model in self.models_[:-1]:
            model.dataset.update(entry_id, (label<=model.rep))
        self.models_[-1].dataset.update(entry_id, label)


    def make_query(self):
        self.ask_qs = np.random.choice(np.where(self.w == np.max(self.w))[0])
        print('qs:', self.ask_qs, self.w)
        ask_id = self.models_[self.ask_qs].make_query()
        return ask_id
