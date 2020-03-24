from sklearn.feature_extraction.text import CountVectorizer
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from settings import USER_COLUMN, ITEM_COLUMN, RATING_COLUMN
import numpy as np
import pandas as pd


def get_movies_features(dataset_path, saved_movies_features=None):
    if saved_movies_features is not None:
        movies_data = pd.read_csv(saved_movies_features, index_col=0)
    else:
        features_extractor = FeaturesExtractor(dataset_path)
        movies_data = features_extractor.run()
        movies_data = movies_data.drop_duplicates(subset=['id'])

    cv = CountVectorizer(min_df=20)
    movies_features = cv.fit_transform(movies_data['combined']).toarray().astype(float)

    return movies_data, movies_features


def get_movie_id_to_feature_mapping(movies_metadata_df):
    mapping = {}
    for i, row in movies_metadata_df.iterrows():
        features = {
            "title": row["title"],
            "id": row["id"],
        }

        mapping[int(row['id'])] = features

    return mapping


def get_weighted_movies_user_features(user_ratings_df, indexer, movies_features):
    user_features = []

    for user_internal_id, user_id in indexer.internal_id_to_user_id_dict.items():
        user_ratings = user_ratings_df[user_ratings_df[USER_COLUMN] == user_id][[ITEM_COLUMN, RATING_COLUMN]].values
        user_rated_movies_id = [indexer.get_movie_internal_id(i) for i in user_ratings[:, 0].astype(int)]
        user_ratings = np.expand_dims(user_ratings[:, 1] / 5.0, axis=1)
        user_rated_movies_features = movies_features[user_rated_movies_id, :]
        user_movies_features = np.sum(np.multiply(user_ratings, user_rated_movies_features), axis=0)
        user_features.append(user_movies_features)

    return np.array(user_features)
