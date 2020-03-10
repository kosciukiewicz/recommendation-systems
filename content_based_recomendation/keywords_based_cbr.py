from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KeywordsBasedCbr:

    def __init__(self):
        self.similarity = None

    def fit(self, data, min_df=3):
        cv = CountVectorizer(min_df=min_df)
        x = cv.fit_transform(data)
        self.similarity = cosine_similarity(x)

    def predict(self, user_id, n):
        pass

    def movie_based_recommendation(self, movie_id, n):
        return np.argsort(-self.similarity[movie_id])[1:n + 1]