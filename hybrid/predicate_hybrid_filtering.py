import itertools
from hybrid.hybrid_filtering import HybridFiltering


class PredicateHybridFiltering(HybridFiltering):
    def __init__(self, filterings, filter_index_mapper, max_len=None):
        super(PredicateHybridFiltering, self).__init__(filterings, max_len)
        self.filter_index_mapper = filter_index_mapper

    def get_top(self, user_id, k=10):
        m = self.max_len
        predicts = list(map(lambda f: f.get_top(user_id, m) if hasattr(f, 'get_top') else f.get_recommendations(user_id, m), self.filterings))
        accepted = list(map(lambda e_p: [x for x in e_p[1] if e_p[0] == self.filter_index_mapper(user_id, e_p[1])], enumerate(predicts)))
        predicted = [x for x in itertools.chain.from_iterable(itertools.zip_longest(*accepted)) if x is not None]
        return predicted[:k]
