
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy, math

class ucb1(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(ucb1, self).__init__(*args, **kwargs)
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
        self.ucb1_ = self.ucb1()
        self.reward = -1

        clf = copy.deepcopy(self.clf)
        clf.train(*(self.dataset.format_sklearn()))
        cost = clf.score(self.test_set[0], self.test_set[1])
        self.last_cost = cost

    def calc_reward(self, entry_id, label):
        clf = copy.deepcopy(self.clf)
        clf.train(*(self.dataset.format_sklearn()))
        cost = clf.score(self.test_set[0], self.test_set[1])
        reward = self.last_cost - cost
        self.last_cost = cost
        return -cost
        #self.last_reward = cost
        #if reward < 0:
        #    return 0.
        #else:
        #    return reward

    def update(self, entry_id, label):
        self.reward = self.calc_reward(entry_id, label)
        print(self.reward)
        for model in self.models_:
            model.dataset.update(entry_id, (label<=model.rep))

    def upperBound(self, step, numPlays):
        return math.sqrt(2 * math.log(step + 1) / numPlays)

    def ucb1(self):
        arms = self.models_
        n_arms = len(arms)
        t = 0
        n = [0 for i in range(n_arms)]
        emp_sum = [0 for i in range(n_arms)]

        for i in range(n_arms):
            reward = yield i
            emp_sum[i] += reward
            t += 1
            n[i] += 1

        while True:
            ucbs = [emp_sum[i] / n[i] + self.upperBound(t, n[i]) for i in range(n_arms)]
            choice = np.argmax(ucbs)
            print(ucbs)
            reward = yield choice
            n[choice] += 1
            emp_sum[choice] += reward
            t += 1

    def make_query(self):
        if self.reward == -1:
            ask_arm = next(self.ucb1_)
        else:
            ask_arm = self.ucb1_.send(self.reward)
        print('arm:', ask_arm)
        ask_id = self.models_[ask_arm].make_query()
        return ask_id
