from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import numpy as np


class HitRate:
    def __init__(self, n):
        self.n = n
        self.user_col_id = 0
        self.movie_col_id = 1

    def evaluate(self, recommendation_method, ratings):
        hits = 0
        cross_validation = LeaveOneOut()

        for counter, (train, test) in enumerate(cross_validation.split(ratings)):
            test_user = int(ratings[test[0], self.user_col_id])
            recommendation_method.fit(np.delete(ratings, test, axis=0))
            selected_items = recommendation_method.get_recommendations(test_user,
                                                                       self.n)

            test_movie = int(ratings[test[0], self.movie_col_id])

            hit = 1 if test_movie in selected_items else 0
            hits += hit
            if hit > 0:
                print(f'{counter} hits: {hits}')
        return hits

