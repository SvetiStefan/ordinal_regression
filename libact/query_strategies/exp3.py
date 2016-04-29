
from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.base.interfaces import QueryStrategy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import copy, math

class Exp3(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(Exp3, self).__init__(*args, **kwargs)
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

        self.exp3_ = self.exp3()
        self.reward = -1
        self.cost = 1.

        K = len(self.models_)
        self.quota = kwargs.pop('quota', 150)
        #self.gamma = np.min(np.sqrt(np.log(K)*K/(np.e-1)/self.quota))
        self.gamma = 0.5
        print('gamma: ', self.gamma)
        #self.gamma = kwargs.pop('gamma', None)

        self.hist_id = []
        self.hist_lbl = []
        self.hist_proba = []
        self.rewards = []

    def calc_reward(self, entry_id, label):
        #clf = self.clf
        #clf.train(*(self.dataset.format_sklearn()))
        #costs = np.abs(np.array(self.hist_lbl) -
        #                clf.predict(np.array([self.dataset.data[i][0] for i in self.hist_id])))
        ##return np.sum(costs / np.array(self.hist_score))
        ##return (10.-np.mean(costs)) / 10.
        #ret = self.cost - (10.-np.mean(costs)) / 10.
        #self.cost = (10.-np.mean(costs)) / 10.

        #return (ret - np.min(self.rewards)) / np.mean(self.rewards)
        clf = LogisticRegression()
        dataset = self.dataset
        dataset.data[entry_id] = (dataset.data[entry_id][0], None)
        X, y = zip(*dataset.get_labeled_entries())
        clf.train(Dataset(X, y<=(self.ask_qs+1)))
        #clf.train(Dataset(X, y<=((self.ask_qs%9)+1)))
        #ori_prob = clf.predict_real(
        #    np.array([self.dataset.data[i][0] for i in self.hist_id]))[:, 0]
        ori_prob = clf.predict_real(
            np.array([self.dataset.data[i][0] for i in range(len(self.dataset.data))]))[:, 0]

        dataset.data[entry_id] = (dataset.data[entry_id][0], label)
        X, y = zip(*dataset.get_labeled_entries())
        clf.train(Dataset(X, y<=(self.ask_qs+1)))
        #clf.train(Dataset(X, y<=((self.ask_qs%9)+1)))
        #prob = clf.predict_real(
        #    np.array([self.dataset.data[i][0] for i in self.hist_id]))[:, 0]
        prob = clf.predict_real(
            np.array([self.dataset.data[i][0] for i in range(len(self.dataset.data))]))[:, 0]

        ret = np.mean(np.abs(ori_prob-prob))
        self.rewards.append(ret)

        if len(self.rewards) == 1:
            return ret
        else:
            return (ret - np.min(self.rewards)) / (np.max(self.rewards) - np.min(self.rewards))

    def update(self, entry_id, label):
        self.hist_id.append(entry_id)
        self.hist_lbl.append(label)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))

        self.reward = self.calc_reward(entry_id, label)
        #self.reward = sigmoid(self.calc_reward(entry_id, label) - 0.5)
        for model in self.models_:
            model.dataset.update(entry_id, (label<=model.rep))

    def exp3(self):
        n_actions = len(self.models_)
        w = np.array([1.0 for i in range(n_actions)])
        gamma = self.gamma
        while True:
            dist = (1-gamma) * w / np.sum(w) + gamma / n_actions
            #dist = w / np.sum(w)
            choice = np.random.choice(np.arange(n_actions), p=dist)
            reward = yield choice
            print(reward, dist, w, self.rewards)
            xhat = reward / dist[choice]
            w[choice] *= np.exp(gamma * xhat / n_actions)

    def make_query(self):
        if self.reward == -1:
            self.ask_qs = next(self.exp3_)
        else:
            self.ask_qs = self.exp3_.send(self.reward)
        print('qs:', self.ask_qs)
        ask_id = self.models_[self.ask_qs].make_query()
        return ask_id
