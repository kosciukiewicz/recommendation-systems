import pandas as pd
import numpy as np
import os

from settings import PATH_TO_DATA, PATH_TO_PROJECT
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from utils.id_indexer import Indexer
from utils.evaluation.test_train_split import user_leave_on_out

from deep_learning.movie_features_deep_learning_method import MovieFeaturesDeepLearningMethod

user_column = 'userId'
item_column = 'movieId'
rating_column = 'rating'


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
        user_ratings = user_ratings_df[user_ratings_df[user_column] == user_id][[item_column, rating_column]].values
        user_rated_movies_id = [indexer.get_movie_internal_id(i) for i in user_ratings[:, 0].astype(int)]
        user_ratings = np.expand_dims(user_ratings[:, 1] / 5.0, axis=1)
        user_rated_movies_features = movies_features[user_rated_movies_id, :]
        user_movies_features = np.sum(np.multiply(user_ratings, user_rated_movies_features), axis=0)
        user_features.append(user_movies_features)

    return np.array(user_features)


def map_df_to_model_train_input(data_df, movies_features, user_features, indexer):
    data = data_df[[user_column, item_column, rating_column]].values
    return [(user_features[indexer.get_user_internal_id(r[0])],
             movies_features[indexer.get_movie_internal_id(r[1])],
             r[2]) for r in data]


def map_df_to_test_input(data_df, movies_features, user_features, indexer):
    data = data_df[[user_column, item_column, rating_column]].values
    return [(indexer.get_user_internal_id(r[0]),
             indexer.get_movie_internal_id(r[1]),
             r[2]) for r in data]


def main():
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    movies_metadata = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/movies_metadata_clean.csv")
    movie_id_features_dict = get_movie_id_to_feature_mapping(movies_metadata)

    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]

    # train_ratings_df, test_ratings_df = train_test_split(user_ratings_df, train_size=0.8, shuffle=True,
    #                                                     random_state=123)

    train_ratings_df, test_ratings_df = \
        list(user_leave_on_out(user_ratings_df, timestamp_column='timestamp', make_user_folds=False))[0]

    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    movies_data = features_extractor.run()

    cv = CountVectorizer(min_df=10)
    movies_features = cv.fit_transform(movies_data['keywords']).toarray().astype(float)
    indexer = Indexer(user_ids=user_ratings_df[user_column].unique(), movies_ids=movies_data['id'])
    user_features = get_weighted_movies_user_features(train_ratings_df, indexer, movies_features)

    train_data = map_df_to_model_train_input(train_ratings_df, movies_features, user_features, indexer)
    test_data = map_df_to_test_input(test_ratings_df, movies_features, user_features, indexer)

    method = MovieFeaturesDeepLearningMethod()
    # method.fit(train_data, test_data, epochs=20)
    method.load_model(filepath=os.path.join(PATH_TO_PROJECT, "deep_learning", "saved_models", "model.h5"),
                      input_size=movies_features.shape[1])

    for user, movie, rating in test_data[:6]:
        recommendations = method.get_recommendations(user_features[user, :], movies_features, k=10)

        user_rated_movies = user_ratings_df[user_ratings_df[user_column] == indexer.get_user_id(user)] \
            .sort_values(rating_column, ascending=False)[[item_column]]\
            .values.squeeze()

        recommended_movies = [indexer.get_movie_id(movie_internal_id) for movie_internal_id in recommendations]

        print("Rated movies: ")
        for movie_id in user_rated_movies:
            print(movie_id_features_dict[movie_id])

        print("Recommended movies: ")
        for movie_id in recommended_movies:
            print(movie_id_features_dict[movie_id])


if __name__ == '__main__':
    main()
