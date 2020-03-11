from abc import abstractmethod
import interfaces


class HybridFiltering(interfaces.RecommendationMethod):
    def __init__(self, filterings, max_len=None):
        self.filterings = filterings
        self.max_len = max_len

    def fit(self, filterings_args, max_len=None):
        self.max_len = max_len if max_len is not None else self.max_len
        map(lambda x: x.fit(filterings_args), self.filterings)

    @abstractmethod
    def get_recommendations(self, user_id, k=10):
        pass
