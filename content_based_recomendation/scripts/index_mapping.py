import numpy as np


class IndexMapping:

    def __init__(self, movies_maping):
        self.user_mapping = None
        self.movie_mapping = movies_maping
        self.internal_movie_mapping = {v: k for k, v in movies_maping.items()}

        self.user_col_id = 0
        self.movie_col_id = 1
        self.rating_col_id = 2

    def map_external_user_id(self, external_id):
        return self.user_mapping[external_id]

    def map_internal_movie_id(self, internal_id):
        return self.internal_movie_mapping[internal_id]

    def get_users_matrix(self, ratings):
        self.reassign_movie_id(ratings)
        self.reassign_user_id(ratings)
        ratings = self.remove_na_id(ratings)
        return self.generate_user_matrix(ratings), \
               self.generate_rated_mask_matrix(ratings)

    def reassign_ids(self, ratings, mapping, col):
        assignement = np.vectorize(lambda current_id: mapping.get(current_id))
        ratings[:, col] = assignement(ratings[:, col])

    def reassign_movie_id(self, ratings):
        self.reassign_ids(ratings, self.movie_mapping, self.movie_col_id)

    def reassign_user_id(self, ratings):
        self.user_mapping = dict(zip(set(ratings[:, self.user_col_id]),
                                     list(range(len(set(ratings[:, self.user_col_id]))))))
        self.reassign_ids(ratings, self.user_mapping, self.user_col_id)

    def to_matrix(self, ratings, init, get_value):
        matrix = init((int(max(ratings[:, self.user_col_id])) + 1,
                       int(max(ratings[:, self.movie_col_id])) + 1))

        def fill_matrix(row):
            matrix[int(row[self.user_col_id]),
                   int(row[self.movie_col_id])] = get_value(row)

        np.apply_along_axis(fill_matrix, axis=1, arr=ratings)
        return matrix

    def generate_user_matrix(self, ratings):
        return self.to_matrix(ratings, np.zeros,
                              lambda row: row[self.rating_col_id])

    def generate_rated_mask_matrix(self, ratings):
        return self.to_matrix(ratings, np.ones, lambda row: 0)

    def remove_na_id(self, ratings):
        return ratings[~np.isnan(ratings).any(axis=1)]


