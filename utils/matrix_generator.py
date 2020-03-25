import numpy as np

class RatingsBasedMatrixGenerator:
    def __init__(self, indexer):
        self.indexer = indexer

        self.user_col_id = 0
        self.movie_col_id = 1
        self.rating_col_id = 2

    def to_matrix(self, ratings, init, get_value):
        matrix = init((len(self.indexer.internal_id_to_user_id_dict),
                       len(self.indexer.internal_id_to_movie_id_dict)))

        def fill_matrix(row):
            matrix[int(row[self.user_col_id]),
                   int(row[self.movie_col_id])] = get_value(row)

        np.apply_along_axis(fill_matrix, axis=1, arr=ratings)
        return matrix

    def generate_user_matrix(self, ratings):
        return self.to_matrix(ratings, np.zeros,
                              lambda row: row[self.rating_col_id])