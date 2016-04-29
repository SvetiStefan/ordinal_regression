
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy, math

class ModelChange(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(ModelChange, self).__init__(*args, **kwargs)
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
        self.w = [10. for i in range(len(self.models_))]
        self.n_ord = 10

        self.hist_id = []
        self.hist_lbl = []
        self.hist_proba = []
        self.rewards = []

        self.C = 1.0

    def boundary_diff(self, entry_id, label, boundary_num):
        sigmoid = lambda t: 1. / (1. + np.exp(-t))
        clf = LogisticRegression()
        dataset = self.dataset
        dataset.data[entry_id] = (dataset.data[entry_id][0], None)
        X, y = zip(*dataset.get_labeled_entries())
        clf.train(Dataset(X, y<=(boundary_num+1)))
        w = clf.model.coef_[0]
        b = clf.model.intercept_[0]
        xi = dataset.data[entry_id][0]
        #grad = self.C * w + (sigmoid(np.sum(w * xi + b)) - (label <= (boundary_num+1))) * xi
        grad = self.C * b + (sigmoid(np.sum(w * xi + b)) - ((label <= (boundary_num+1))*2 - 1))
        #print(xi, w, b)
        dataset.data[entry_id] = (dataset.data[entry_id][0], label)
        return np.linalg.norm(grad)

    def update_changes(self, entry_id, label):
        #import ipdb; ipdb.set_trace()
        #for i in range(self.n_ord-1):
        #    self.w[i] = self.boundary_diff(entry_id, label, np.int64(i))
        self.w[self.ask_qs] = self.boundary_diff(entry_id, label, np.int64(self.ask_qs))

    def update(self, entry_id, label):
        self.hist_id.append(entry_id)
        self.hist_lbl.append(label)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        self.update_changes(entry_id, label)
        #self.reward = sigmoid(self.calc_reward(entry_id, label) - 0.5)
        for model in self.models_[:-1]:
            model.dataset.update(entry_id, (label<=model.rep))

    def make_query(self):
        self.ask_qs = np.random.choice(np.where(self.w == np.max(self.w))[0])
        print('qs:', self.ask_qs, self.w)
        ask_id = self.models_[self.ask_qs].make_query()
        return ask_id
