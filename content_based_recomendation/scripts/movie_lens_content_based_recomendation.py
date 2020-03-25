from settings import PATH_TO_DATA
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from content_based_recomendation.weigted_rating_cbr import WeightedRatingCbr
from content_based_recomendation.keywords_based_cbr import KeywordsBasedCbr
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)


def run_kbcbr_example(data, n=3):
    keyword_cbr = KeywordsBasedCbr()
    keyword_cbr.fit(data['combined'])

    df = data.copy()
    df['most_similar'] = [[df.iloc[movie_id]['title']
                           for movie_id in keyword_cbr.predict(movie_id, n)]
                           for movie_id in data.index]
    df['avg_similarity'] = [np.mean(keyword_cbr.get_highest_similarities(movie_id, n))
                            for movie_id in data.index]
    df = df.sort_values(by=['avg_similarity'])

    return df[['title', 'avg_similarity', 'most_similar', 'combined']]


def main():
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()

    print('Keywords based CBR method example')
    print(run_kbcbr_example(data).tail(30))


if __name__ == '__main__':
    main()