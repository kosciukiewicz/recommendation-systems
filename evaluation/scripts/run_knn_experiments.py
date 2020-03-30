from random import random

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from deep_learning.utils import get_movie_id_to_feature_mapping
from evaluation.scripts.run_clustering_experiments import create_kmeans_item_based_input_df
from knn.knn_collaborative_filtering import KnnCollaborativeFiltering
from knn.knn_item_based import KnnItemBased
from utils.evaluation.metrics import hit_rate
from utils.evaluation.test_train_split import user_leave_on_out
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from utils.id_indexer import Indexer
from settings import PATH_TO_DATA

user_column = 'userId'
item_column = 'movieId'
rating_column = 'rating'


def prepare_popularities(movies_metadata, movies_data):
    popularities = list()
    for index, row in movies_data.iterrows():
        popularities.append(movies_metadata[movies_metadata["id"] == row["id"]]["popularity"].values[0])
    if len(popularities) != len(movies_data):
        raise ValueError("Len of popularities must be equal to len of movies_data")
    return popularities


def knn_item_based():
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
    for metric in ['cosine', 'minkowski']:
        for threshold in [4.0, None]:

            method = KnnItemBased(n_neighbours=50, metric=metric)
            method.fit(movies_features, indexer)

            df_dataset = create_kmeans_item_based_input_df(user_ratings_df, movies_features, indexer,
                                                           timestamp_column="timestamp",
                                                           rating_threshold=threshold)

            print("Testing...")
            iterations = 0
            all_hits = 0

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

            results.append([metric, threshold, all_hits])

    for row in results:
        print(row)


def knn_collaborative_filtering():
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    movies_metadata = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/movies_metadata_clean.csv")
    movie_id_features_dict = get_movie_id_to_feature_mapping(movies_metadata)
    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]

    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    movies_data = features_extractor.run()
    movies_data = movies_data.drop_duplicates(["id"])

    indexer = Indexer(user_ids=user_ratings_df[user_column].unique(), movies_ids=movies_data['id'])

    results = list()
    metric = 'cosine'   # 'euclidean'
    for threshold in [None, 4.0]:

        method = KnnCollaborativeFiltering(indexer, n_neighbors=10, metric=metric)

        print("Testing...")
        iterations = 0
        all_hits = 0

        for train_df, test_df in user_leave_on_out(user_ratings_df, timestamp_column="timestamp", rating_threshold=threshold):
            print(iterations)
            train_ratings = train_df.values[:, :3]
            user_id, item_id, rating = test_df.values[:, :3][0]
            method.fit(train_ratings)
            pred_ids = method.get_recommendations(user_id, top_n=30)

            print("Recommended movies: ")
            for movie_id in pred_ids:
                print(movie_id_features_dict[indexer.get_movie_id(movie_id)])

            hits = hit_rate(gt_items_idx=[item_id.astype(int)], predicted_items_idx=pred_ids)

            all_hits += hits
            iterations += 1

        if all_hits > 0:
            print(f"{method.__class__}: {all_hits}/{iterations}")
            print(f"Percentage-wise: {all_hits / iterations}")
        print(f"Total hits: {all_hits}")
        print(f"Total iterations: {iterations}")

        results.append([metric, threshold, all_hits])

    for row in results:
        print(row)


if __name__ == '__main__':
    # knn_item_based()
    knn_collaborative_filtering()