import itertools
from hybrid.hybrid_filtering import HybridFiltering
import numpy as np


class AverageHybridFiltering(HybridFiltering):
    def __init__(self, filterings, max_len=None):
        super(AverageHybridFiltering, self).__init__(filterings, max_len)

    def get_top(self, user_id, k=10):
        m = self.max_len
        predicts = list(map(lambda f: list(f.get_top(user_id, m)) if hasattr(f, 'get_top') else list(f.get_recommendations(user_id, m)), self.filterings))
        unique = np.unique([x for x in itertools.chain.from_iterable(itertools.zip_longest(*predicts)) if x is not None]).tolist()
        positions = np.array([[predicts[i].index(x) if x in predicts[i] else np.NaN for x in unique] for i in range(len(predicts))])
        avg = np.nanmean(positions, axis=0)
        inds = np.argsort(avg)
        predicted = list(np.array(unique)[inds])
        return predicted[:k]
