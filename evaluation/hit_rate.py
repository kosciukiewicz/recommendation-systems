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

        print(recommendation_method)
        for train, test in cross_validation.split(ratings):
            test_user = int(ratings[test[0], self.user_col_id])
            recommendation_method.fit(np.delete(ratings, test, axis=0))
            selected_items = recommendation_method.get_recommendations(test_user,
                                                                       self.n)

            test_movie = int(ratings[test[0], self.movie_col_id])
            hits += 1 if test_movie in selected_items else 0
            print(hits)
        return hits

    # def is_hit(self, ratings, user_id, selected_items):
    #     hits = ratings[ratings[:, self.user_col_id] == user_id][:, self.movie_col_id]
    #     return len(set.intersection(set(selected_items),
    #                                 set(hits)))
