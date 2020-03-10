from sklearn.model_selection import LeaveOneOut
from user_settings import PATH_TO_DATA
import pandas as pd

class HitRate:
    def __init__(self, n):
        self.n = n

    def evaluate(self, recommendation_method, user_items_ratings):
        pass

    def is_hit(self, user_items_ratings, user_id, selected_items):
        pass



