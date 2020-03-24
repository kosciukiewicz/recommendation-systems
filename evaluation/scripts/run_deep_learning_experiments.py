import deep_learning.utils as deep_learning_utils
import pandas as pd
import numpy as np
import os

from collaborative_filtering.neural_collaborative_filtering import NeuralCollaborativeFiltering
from deep_learning.movie_features_deep_learning_method import MovieFeaturesDeepLearningMethod
from sklearn.feature_extraction.text import CountVectorizer
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from utils.id_indexer import Indexer
from utils.evaluation.test_train_split import user_leave_on_out
from utils.evaluation.metrics import hit_rate
from settings import PATH_TO_DATA, PATH_TO_PROJECT

user_column = 'userId'
item_column = 'movieId'
rating_column = 'rating'


def get_movies_model_features(all_users_ratings_df, train_ratings_df, movies_feature_data_path, saved_movies_features):
    movies_data_df, movies_features = deep_learning_utils.get_movies_features(movies_feature_data_path,
                                                                              saved_movies_features)
    movies_data_df.to_csv(f"{PATH_TO_DATA}/generated/movies_data_df.csv")
    indexer = Indexer(user_ids=all_users_ratings_df[user_column].unique(), movies_ids=movies_data_df['id'].unique())

    user_features = deep_learning_utils.get_weighted_movies_user_features(train_ratings_df, indexer, movies_features)

    return movies_features, user_features, indexer


def map_df_to_model_input(data_df):
    data = data_df[[user_column, item_column, rating_column]].values
    return [(int(r[0]),
             int(r[1]),
             r[2] / 5) for r in data]


def load_neucf_model(saved_model_filename, indexer, train_data):
    method = NeuralCollaborativeFiltering(indexer=indexer, n_factors=64, model_name="neucf_64_wo_threshold")
    method.load_model(
        filepath=os.path.join(PATH_TO_PROJECT, "collaborative_filtering", "saved_models", saved_model_filename),
        train_user_item_ratings=train_data)
    return method


def load_movies_features_model(saved_model_filename, indexer, movies_features, user_features):
    method = MovieFeaturesDeepLearningMethod(indexer=indexer, movies_features=movies_features,
                                             user_features=user_features, model_name="movies_features_wo_threshold")
    method.load_model(
        filepath=os.path.join(PATH_TO_PROJECT, "deep_learning", "saved_models", saved_model_filename))
    return method


def main():
    dataset_path = os.path.join(os.sep, PATH_TO_DATA, "raw/the-movies-dataset")
    path_to_saved_features = os.path.join(os.sep, PATH_TO_DATA, "generated/movies_data_df.csv")
    user_ratings_df = pd.read_csv(os.path.join(dataset_path, "ratings_small.csv"))
    movies_metadata = pd.read_csv(os.path.join(dataset_path, "movies_metadata_clean.csv"))
    movie_id_features_dict = deep_learning_utils.get_movie_id_to_feature_mapping(movies_metadata)

    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]

    train_ratings_df, test_ratings_df = \
        list(user_leave_on_out(user_ratings_df, timestamp_column="timestamp", make_user_folds=False, rating_threshold=4.0))[0]

    movies_features, user_features, indexer = get_movies_model_features(user_ratings_df, train_ratings_df, dataset_path,
                                                                        path_to_saved_features)

    train_data = map_df_to_model_input(train_ratings_df)
    test_data = map_df_to_model_input(test_ratings_df)

    # neucf_loader = lambda: load_neucf_model("neucf_64_wo_threshold.h5", indexer, train_data)
    # movies_feature_model_loader = lambda: load_movies_features_model("movies_features_wo_threshold.h5", indexer,
    #                                                                  movies_features, user_features)

    neucf_loader = lambda: load_neucf_model("neucf_64_w_4_0_threshold.h5", indexer, train_data)
    movies_feature_model_loader = lambda: load_movies_features_model("movies_features_w_4_0_threshold.h5", indexer,
                                                                     movies_features, user_features)

    models = [
        neucf_loader(),
        movies_feature_model_loader()
    ]

    n = 30

    results = {}

    for model in models:
        iterations = 0
        all_hits = 0
        for user, movie, rating in test_data:
            print(iterations)
            recommendations = model.get_recommendations(user_id=user)
            user_rated_movies = train_ratings_df[train_ratings_df[user_column] == user] \
                .sort_values(rating_column, ascending=False)[[item_column]] \
                .values.squeeze()

            recommended_movies = [movie_id for movie_id in recommendations if movie_id not in user_rated_movies][:n]
            hits = hit_rate(gt_items_idx=[movie], predicted_items_idx=recommended_movies)

            all_hits += hits
            iterations += 1

            if hits > 0:
                print(f"{model.__class__}: {all_hits}/{iterations}")

        results[model.__class__] = all_hits
        print(results)


if __name__ == '__main__':
    main()
