import pandas as pd
import os
import deep_learning.utils as deep_learning_utils
import pickle as plk

from settings import PATH_TO_DATA, PATH_TO_PROJECT
from utils.id_indexer import Indexer
from utils.evaluation.test_train_split import user_leave_on_out

from deep_learning.movie_features_deep_learning_method import MovieFeaturesDeepLearningMethod
from collaborative_filtering.neural_collaborative_filtering import NeuralCollaborativeFiltering

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


def main():
    dataset_path = os.path.join(os.sep, PATH_TO_DATA, "raw/the-movies-dataset")
    path_to_saved_features = os.path.join(os.sep, PATH_TO_DATA, "generated/movies_data_df.csv")
    user_ratings_df = pd.read_csv(os.path.join(dataset_path, "ratings_small.csv"))
    movies_metadata = pd.read_csv(os.path.join(dataset_path, "movies_metadata_clean.csv"))
    movie_id_features_dict = deep_learning_utils.get_movie_id_to_feature_mapping(movies_metadata)

    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]

    train_ratings_df, test_ratings_df = \
    list(user_leave_on_out(user_ratings_df, timestamp_column="timestamp", make_user_folds=False))[0]

    movies_features, user_features, indexer = get_movies_model_features(user_ratings_df, train_ratings_df, dataset_path,
                                                                        path_to_saved_features)

    train_data = map_df_to_model_input(train_ratings_df)
    test_data = map_df_to_model_input(test_ratings_df)

    #method = NeuralCollaborativeFiltering(indexer=indexer, n_factors=64, model_name="neucf_64_w_4_0_threshold")

    method = MovieFeaturesDeepLearningMethod(indexer=indexer, movies_features=movies_features,
                                              user_features=user_features, model_name="movies_features_w_4_0_threshold")

    method.fit(train_data, test_data, batch_size=25, epochs=50, early_stopping=5)

    # method.load_model(filepath=os.path.join(PATH_TO_PROJECT, "collaborative_filtering", "saved_models", "neucf_64_w_4_0_threshold.h5"), train_user_item_ratings=train_data)
    # method.load_model(filepath=os.path.join(PATH_TO_PROJECT, "deep_learning", "saved_models", "movies_features_w_4_0_threshold.h5"))

    # for user, movie, rating in test_data[:10]:
    #     recommendations = method.get_recommendations(user_id=user)
    #
    #     user_rated_movies = train_ratings_df[train_ratings_df[user_column] == user] \
    #         .sort_values(rating_column, ascending=False)[[item_column]] \
    #         .values.squeeze()
    #
    #     recommended_movies = [movie_internal_id for movie_internal_id in recommendations if
    #                           movie_internal_id not in user_rated_movies][:10]
    #
    #     print("Rated movies: ")
    #     for movie_id in user_rated_movies:
    #         print(movie_id_features_dict[movie_id])
    #
    #     print("Recommended movies: ")
    #     for movie_id in recommended_movies:
    #         print(movie_id_features_dict[movie_id])
    #
    #     print("Test movie rating: ")
    #     print(movie_id_features_dict[movie], rating)


if __name__ == '__main__':
    main()
