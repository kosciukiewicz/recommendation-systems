from evaluation.hit_rate import HitRate
from content_based_recomendation.weigted_rating_cbr import WeightedRatingCbr
from settings import PATH_TO_DATA
from content_based_recomendation.scripts.movie_lens_features_extractor import FeaturesExtractor
from collaborative_filtering.memory_based_collaborative_filtering import MemoryBasedCollaborativeFiltering
from collaborative_filtering.svd_collaborative_filtering import SVDCollaborativeFiltering
import pandas as pd
import os


def main():
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    user_column = 'userId'
    item_column = 'movieId'

    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small_clean.csv'))

    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()

    movie_mapping = dict(zip(data['id'].tolist(), data.index.astype(int)))

    user_ids = ratings[user_column].unique()
    movie_ids = ratings[item_column].unique()

    hit_rate_ns = [30]
    methods = [MemoryBasedCollaborativeFiltering(user_ids, movie_ids),
               WeightedRatingCbr(data['combined'], movie_mapping),
               SVDCollaborativeFiltering(user_ids, movie_ids)]

    for n in hit_rate_ns:
        hit_rate = HitRate(n)
        for method in methods:
            print(type(method))
            result = hit_rate.evaluate(method, ratings.values[:, :3])
            print(f'Final result: {result}')


if __name__ == '__main__':
    main()