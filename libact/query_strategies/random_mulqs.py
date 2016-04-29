
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy, random

class RandMulQs(QueryStrategy):
    def __init__(self, *args, **kwargs):
        super(RandMulQs, self).__init__(*args, **kwargs)

        self.models_ = kwargs.pop('models', None)
        if self.models_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )
        elif not self.models_:
            raise ValueError("models list is empty")

    def update(self, entry_id, label):
        for model in self.models_:
            model.dataset.update(entry_id, (label<=model.rep))

    def make_query(self):
        q = random.randint(0, len(self.models_)-1)
        return self.models_[q].make_query()



