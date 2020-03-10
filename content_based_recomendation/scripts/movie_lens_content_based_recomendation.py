from settings import PATH_TO_DATA
from content_based_recomendation.scripts.movie_lens_features_extractor import FeaturesExtractor
from content_based_recomendation.keywords_based_cbr import KeywordsBasedCbr
from content_based_recomendation.weigted_rating_cbr import WeightedRatingCbr
import numpy as np
import pandas as pd
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)


def movie_id(data, title):
    return data.index[data['title'] == title].tolist()[0]


def filter_ratings(dataset_path, data):
    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small.csv'))
    ratings = ratings[ratings['movieId'].isin(data['id'])]
    ratings.to_csv(os.path.join(dataset_path, 'ratings_small_clean.csv'),
                  index=False)


def main():
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()

    # keyword_cbr = KeywordsBasedCbr()
    # keyword_cbr.fit(data['combined'])
    # ids = keyword_cbr.movie_based_recommendation(movie_id(data, 'Star Wars'), 5)
    # print(data.iloc[ids]['title'])

    # weighted_rating_cbr = WeightedRatingCbr(data['combined'])
    # movies_mapping, users_mapping, users_matrix, ratings = load_rating(
    #     dataset_path, data)
    # weighted_rating_cbr.fit(users_matrix)
    # print(data.iloc[weighted_rating_cbr.predict(0, 5)]['title'])

    movie_mapping = dict(zip(data['id'].tolist(), data.index.astype(int)))
    weighted_rating_cbr = WeightedRatingCbr(data['combined'], movie_mapping)

    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small.csv'))

    # weighted_rating_cbr.fit(ratings.values)
    #
    # rated_movies = ratings[ratings['userId'] == 1]['movieId']
    # print(data[data['id'].isin(rated_movies)][['id', 'title', 'combined']])
    # # print(data.iloc[weighted_rating_cbr.get_recommendations(0, 10)][['id', 'title']])
    # #
    # recommendations = weighted_rating_cbr.get_recommendations(1, 10)
    # print(data[data['id'].isin(recommendations)][['id', 'title', 'combined']])

    # hr = HitRate(10)
    # hr.evaluate(weighted_rating_cbr, ratings.values, users_matrix)
    filter_ratings(dataset_path, data)


if __name__ == '__main__':
    main()