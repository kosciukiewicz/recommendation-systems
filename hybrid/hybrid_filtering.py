class HybridFiltering:
    def __init__(self, filterings, max_len=None):
        self.filterings = filterings
        self.max_len = max_len

    def fit(self, filterings_args, max_len):
        self.max_len = max_len
        map(lambda f, args: f.fit(*args), zip(self.filterings, filterings_args))

    def get_top(self, user_id, k=10):
        return None
