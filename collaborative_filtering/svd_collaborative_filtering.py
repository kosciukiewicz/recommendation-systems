import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from interfaces import RecommendationMethod


def create_id_vocab(data):
    id_to_data_id_vocab = {}

    id = 0
    for i in data:
        if i not in id_to_data_id_vocab.values():
            id_to_data_id_vocab[id] = i
            id += 1

    return id_to_data_id_vocab, {v: k for k, v in id_to_data_id_vocab.items()}


class SVDCollaborativeFiltering(RecommendationMethod):
    def __init__(self, users_ids, items_ids):
        self.id_to_user_id_vocab, self.user_id_to_id_vocab = create_id_vocab(users_ids)
        self.id_to_item_id_vocab, self.item_id_to_id_vocab = create_id_vocab(items_ids)
        self.user_ratings = None
        self.u = None
        self.s = None
        self.vt = None

    def fit(self, user_items_ratings, n_factors=None):
        self.user_ratings = np.zeros((len(self.id_to_user_id_vocab.keys()), len(self.id_to_item_id_vocab.keys())))

        for i in tqdm(range(len(user_items_ratings))):
            user, item, rating = user_items_ratings[i]
            user_index = self.user_id_to_id_vocab[int(user)]
            item_index = self.item_id_to_id_vocab[int(item)]

            self.user_ratings[user_index, item_index] = rating

        if n_factors is None:
            n_factors = min(self.user_ratings.shape) - 1

        sparse_user_ratings = csc_matrix(self.user_ratings, dtype=float)
        self.u, self.s, self.vt = svds(sparse_user_ratings, k=n_factors)

    def predict(self, user_id, item_id):
        return self.u[self.user_id_to_id_vocab[user_id], :] \
            .dot(np.diag(self.s)) \
            .dot(self.vt[:, self.item_id_to_id_vocab[item_id]])

    def get_recommendations(self, user_id, k):
        recommendations = self.u[self.user_id_to_id_vocab[user_id], :] \
            .dot(np.diag(self.s)) \
            .dot(self.vt)
        non_rated_user_movies = self.user_ratings[self.user_id_to_id_vocab[user_id], :] == 0

        recommendations_idx = np.argsort(recommendations)[::-1]
        return [self.id_to_item_id_vocab[i] for i in recommendations_idx[:k] if non_rated_user_movies[i]]
