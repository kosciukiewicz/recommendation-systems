from settings import PATH_TO_DATA
from content_based_recomendation.scripts.movie_lens_features_extractor import FeaturesExtractor
from content_based_recomendation.keywords_based_cbr import KeywordsBasedCbr
from content_based_recomendation.weigted_rating_cbr import WeightedRatingCbr
import numpy as np
import pandas as pd
import os


def movie_id(data, title):
    return data.index[data['title'] == title].tolist()[0]

#TODO

def reassign_movie_id(data, ratings):
    movie_mapping = dict(zip(data['id'].tolist(), data.index.astype(int)))
    ratings['movieId'] = ratings['movieId'].map(movie_mapping)
    return movie_mapping


def reassign_user_id(data, ratings):
    user_mapping = dict(zip(set(ratings['userId']),
                            list(range(len(set(ratings.userId))))))
    ratings['userId'] = ratings['userId'].map(user_mapping)
    return user_mapping


def remove_na_id(ratings, cols_id):
    for col in cols_id:
        ratings = ratings[ratings[col].notna()]
        ratings[col] = ratings[col].astype(int)
    return ratings.reset_index(drop=True)


def df_to_matrix(df, col_index1, col_index2, col_value):
    matrix = np.zeros((max(df[col_index1].tolist()) + 1,
                       max(df[col_index2].tolist()) + 1))

    def fill_matrix(row):
        matrix[int(row[col_index1]),
               int(row[col_index2])] = row[col_value]

    df.apply(fill_matrix, axis=1)
    return matrix


def load_rating(dataset_path, data):
    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small.csv'))
    movies_mapping = reassign_movie_id(data, ratings)
    users_mapping = reassign_user_id(data, ratings)
    ratings = remove_na_id(ratings, ['movieId', 'userId'])
    users_matrix = df_to_matrix(ratings, 'userId', 'movieId', 'rating')
    return movies_mapping, users_mapping, users_matrix

def main():
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()

    keyword_cbr = KeywordsBasedCbr()
    keyword_cbr.fit(data['combined'])
    ids = keyword_cbr.movie_based_recommendation(movie_id(data, 'Star Wars'), 5)
    print(data.iloc[ids]['title'])

    weighted_rating_cbr = WeightedRatingCbr()
    movies_mapping, users_mapping, users_matrix = load_rating(
        dataset_path, data)
    weighted_rating_cbr.fit(data['combined'], users_matrix)
    print(data.iloc[weighted_rating_cbr.predict(0, 5)]['title'])


if __name__ == '__main__':
    main()