from random import random

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from clustering.kmeans_item_based import KmeansItemBased
from deep_learning.utils import get_movie_id_to_feature_mapping
from utils.evaluation.metrics import hit_rate
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from utils.id_indexer import Indexer
from settings import PATH_TO_DATA

user_column = 'userId'
item_column = 'movieId'
rating_column = 'rating'


# def get_user_features_on_movies_features(user_ratings_df, indexer, movies_features):
#     user_features = []
# 
#     for user_internal_id, user_id in indexer.internal_id_to_user_id_dict.items():
#         user_ratings = user_ratings_df[user_ratings_df[user_column] == user_id][[item_column, rating_column]].values
#         user_rated_movies_id = [indexer.get_movie_internal_id(i) for i in user_ratings[:, 0].astype(int)]
#         user_ratings = np.expand_dims(user_ratings[:, 1] / 5.0, axis=1)
#         user_rated_movies_features = movies_features[user_rated_movies_id, :]
#         user_movies_features = np.multiply(user_ratings, user_rated_movies_features)
#         user_movies_features[user_movies_features == 0] = np.nan
#         user_movies_features = np.nan_to_num(np.nanmean(user_movies_features, axis=0))
#         # user_movies_features[user_movies_features == 0] = 0.5
#         # user_movies_features = np.mean(user_movies_features, axis=0)
# 
#         user_features.append(user_movies_features)
# 
#     return np.array(user_features)
# 
# 
# def kmeans_user_based():
#     user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
#     movies_metadata = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/movies_metadata_clean.csv")
#     movie_id_features_dict = get_movie_id_to_feature_mapping(movies_metadata)
#     user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]
# 
#     dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
#     features_extractor = FeaturesExtractor(dataset_path)
#     movies_data = features_extractor.run()
#     movies_data = movies_data.drop_duplicates(["id"])
# 
#     cv = CountVectorizer(min_df=3)
#     movies_features = cv.fit_transform(movies_data['combined']).toarray().astype(float)
#     indexer = Indexer(user_ids=user_ratings_df[user_column].unique(), movies_ids=movies_data['id'])
# 
#     method = KmeansUserBased(movies_features=movies_features)
# 
#     for train_df, test_df in user_leave_on_out(user_ratings_df, timestamp_column="timestamp"):
#         user_features = get_user_features_on_movies_features(train_df, indexer, movies_features)
# 
#         # train_data = map_df_to_model_input(train_df, movies_features, user_features, indexer)
#         test_data = map_df_to_model_input(test_df, movies_features, user_features, indexer)
# 
#         method.fit(user_features)
# 
#         for i, (index, row) in enumerate(test_df.iterrows()):
#             user, movie, rating = test_data[i]
#             recommendations = method.get_recommendations(user.reshape(1, -1), top_n=10)
# 
#             user_rated_movies = user_ratings_df[user_ratings_df[user_column] == row[user_column]] \
#                 .sort_values(rating_column, ascending=False)[[item_column]] \
#                 .values.squeeze()
# 
#             user_rated_movies_ratings = user_ratings_df[user_ratings_df[user_column] == row[user_column]] \
#                 .sort_values(rating_column, ascending=False)[[rating_column]] \
#                 .values.squeeze()
# 
#             recommended_movies = [indexer.get_movie_id(movie_internal_id) for movie_internal_id in recommendations]
# 
#             print(f"Test movie: {movie_id_features_dict[row[item_column]]}, rating: {row[rating_column]}")
# 
#             print("Rated movies: ")
#             for movie_id, rating in zip(user_rated_movies, user_rated_movies_ratings):
#                 print(movie_id_features_dict[movie_id], f"rating: {rating}")
# 
#             print("Recommended movies: ")
#             for movie_id in recommended_movies:
#                 print(movie_id_features_dict[movie_id])


def create_kmeans_item_based_input_df(user_ratings_df, movies_features, indexer, timestamp_column=None,
                                      rating_threshold=None):
    users_features = list()
    train_movies_ids = list()
    train_ratings = list()
    test_movies_ids = list()
    test_ratings = list()

    user_ids = user_ratings_df[user_column].unique()
    test_indices = []

    for user_id in tqdm(user_ids):
        user_ratings = user_ratings_df[user_ratings_df[user_column] == user_id]

        if rating_threshold is not None:
            thresholded_user_ratings = user_ratings[user_ratings[rating_column] >= rating_threshold]

            while len(thresholded_user_ratings) < 2:
                rating_threshold -= 0.5
                thresholded_user_ratings = user_ratings[user_ratings[rating_column] >= rating_threshold]

            user_ratings = thresholded_user_ratings

        if timestamp_column is not None:
            user_ratings = user_ratings.sort_values(by=timestamp_column)
            test_index = len(user_ratings) - 1
        else:
            test_index = random.randint(0, len(user_ratings) - 1)

        indices = user_ratings.index.values
        test_indices.append(indices[test_index])

        users_feature_matrix = list()
        train_movie_id = list()
        train_rating = list()
        for index in indices[:-1]:
            movie_id = user_ratings_df[item_column][index]
            train_movie_id.append(movie_id)
            users_feature_matrix.append(movies_features[indexer.get_movie_internal_id(movie_id)])
            train_rating.append(user_ratings_df[rating_column][index])
        users_feature_matrix = np.asarray(users_feature_matrix)
        users_features.append(users_feature_matrix)
        train_movies_ids.append(train_movie_id)
        train_ratings.append(train_rating)
        test_movies_ids.append(user_ratings_df[item_column][indices[test_index]])
        test_ratings.append(user_ratings_df[rating_column][indices[test_index]])

    df = pd.DataFrame()
    df["user_id"] = user_ids
    df["user_matrix"] = users_features
    df["train_movie_ids"] = train_movies_ids
    df["train_ratings"] = train_ratings
    df["test_movie_ids"] = test_movies_ids
    df["test_ratings"] = test_ratings

    return df


def prepare_popularities(movies_metadata, movies_data):
    popularities = list()
    for index, row in movies_data.iterrows():
        popularities.append(movies_metadata[movies_metadata["id"] == row["id"]]["popularity"].values[0])
    if len(popularities) != len(movies_data):
        raise ValueError("Len of popularities must be equal to len of movies_data")
    return popularities


def kmeans_item_based():
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    movies_metadata = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/movies_metadata_clean.csv")
    movie_id_features_dict = get_movie_id_to_feature_mapping(movies_metadata)
    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]

    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    movies_data = features_extractor.run()
    movies_data = movies_data.drop_duplicates(["id"])

    cv = CountVectorizer(min_df=3)
    movies_features = cv.fit_transform(movies_data['combined']).toarray().astype(float)
    indexer = Indexer(user_ids=user_ratings_df[user_column].unique(), movies_ids=movies_data['id'])

    results = list()
    # for n_centroids in [5, 10, 20, 50, 100]:
    n_centroids = 50
    method = KmeansItemBased(cluster_selection="random",
                             item_selection="popularity")
    print("Fitting kmeans...")
    method.fit(movies_features, n_clusters=n_centroids,
               popularity=prepare_popularities(movies_metadata, movies_data))

    df_dataset = create_kmeans_item_based_input_df(user_ratings_df, movies_features, indexer,
                                                   timestamp_column="timestamp",
                                                   rating_threshold=None)

    print("Testing...")
    iterations = 0
    all_hits = 0

    for i in range(10):
        for index, row in tqdm(df_dataset.iterrows()):
            user_id = row["user_id"]
            user_matrix = row["user_matrix"]
            train_movie_ids = row["train_movie_ids"]
            train_ratings = row["train_ratings"]
            test_movie_id = row["test_movie_ids"]
            test_rating = row["test_ratings"]

            recommendations = method.get_recommendations(user_matrix, train_movie_ids, train_ratings, top_n=30)

            user_rated_movies = user_ratings_df[user_ratings_df[user_column] == user_id] \
                .sort_values(rating_column, ascending=False)[[item_column]] \
                .values.squeeze()

            user_rated_movies_ratings = user_ratings_df[user_ratings_df[user_column] == user_id] \
                .sort_values(rating_column, ascending=False)[[rating_column]] \
                .values.squeeze()

            recommended_movies = [indexer.get_movie_id(movie_internal_id) for movie_internal_id in
                                  recommendations]

            print(f"Test movie: {movie_id_features_dict[test_movie_id]}, rating: {test_rating}")

            print("Rated movies: ")
            for movie_id, rating in zip(user_rated_movies, user_rated_movies_ratings):
                print(movie_id_features_dict[movie_id], f"rating: {rating}")

            print("Recommended movies: ")
            for movie_id in recommended_movies:
                print(movie_id_features_dict[movie_id])

            hits = hit_rate(gt_items_idx=[test_movie_id], predicted_items_idx=recommendations)

            all_hits += hits
            iterations += 1

    if all_hits > 0:
        print(f"{method.__class__}: {all_hits}/{iterations}")
        print(f"Percentage-wise: {all_hits / iterations}")
    print(f"Total hits: {all_hits}")
    print(f"Total iterations: {iterations}")

    results.append([n_centroids, None, all_hits / 10.0])

    for row in results:
        print(row)


if __name__ == '__main__':
    # kmaens_user_based()
    kmeans_item_based()
