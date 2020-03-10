import os

from sklearn.model_selection import train_test_split
from collaborative_filtering.memory_based_collaborative_filtering import MemoryBasedCollaborativeFiltering
from content_based_recomendation.scripts.movie_lens_features_extractor import FeaturesExtractor
from content_based_recomendation.weigted_rating_cbr import WeightedRatingCbr
from hybrid.average_hybrid_filtering import AverageHybridFiltering
from hybrid.predicate_hybrid_filtering import PredicateHybridFiltering
from settings import PATH_TO_DATA
import pandas as pd


def create_cf():
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    user_column = 'userId'
    item_column = 'movieId'
    rating_column = 'rating'

    ratings_samples = user_ratings_df[[user_column, item_column, rating_column]].values
    ratings_samples = [(int(r[0]), int(r[1]), r[2]) for r in ratings_samples]

    train_ratings, test_ratings = train_test_split(ratings_samples, train_size=0.8, shuffle=True, random_state=123)

    method = MemoryBasedCollaborativeFiltering

    cf = method(users_ids=user_ratings_df[user_column].unique(),
                items_ids=user_ratings_df[item_column].unique())

    cf.fit(user_items_ratings=train_ratings)

    return cf


def create_cbf():
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()

    movie_mapping = dict(zip(data['id'].tolist(), data.index.astype(int)))
    weighted_rating_cbr = WeightedRatingCbr(data['combined'], movie_mapping)

    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small.csv'))

    weighted_rating_cbr.fit(ratings.values)

    return weighted_rating_cbr


def main():
    cf = create_cf()
    cbf = create_cbf()

    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small.csv'))
    ratings_per_user = ratings.groupby('userId').sum()

    # decider = lambda userId, itemId: 0 if userId <= 1 else 1
    # # 100 powinno wystarczyc ale moze trzeba zamieniac na wiecej az do len(ratings_per_user) ale to kosztuje czas
    # hf = PredicateHybridFiltering([cf, cbf], decider, 100)

    hf = AverageHybridFiltering([cf, cbf], len(ratings_per_user))

    recomendations = hf.get_top(1, 10)
    print(recomendations)


if __name__ == '__main__':
    main()
