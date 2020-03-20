from utils.evaluation.metrics import hit_rate
from content_based_recomendation.weigted_rating_cbr import WeightedRatingCbr
from settings import PATH_TO_DATA
from utils.features_extraction.movie_lens_features_extractor import FeaturesExtractor
from collaborative_filtering.memory_based_collaborative_filtering import MemoryBasedCollaborativeFiltering
from collaborative_filtering.svd_collaborative_filtering import SVDCollaborativeFiltering
from utils.evaluation.test_train_split import user_leave_on_out
import pandas as pd
import os


def main():
    dataset_path = f'{PATH_TO_DATA}/raw/the-movies-dataset'
    user_column = 'userId'
    item_column = 'movieId'

    ratings = pd.read_csv(os.path.join(dataset_path, 'ratings_small_clean.csv'))
    ratings_per_user = ratings.groupby('userId').sum()

    features_extractor = FeaturesExtractor(dataset_path)
    data = features_extractor.run()
    movie_mapping = dict(zip(data['id'].tolist(), data.index.astype(int)))

    user_ids = ratings[user_column].unique()
    movie_ids = ratings[item_column].unique()

    wcr_factory = lambda: WeightedRatingCbr(data['combined'], movies_mapping=movie_mapping)

    methods = [
        #MemoryBasedCollaborativeFiltering(user_ids, movie_ids),
        SVDCollaborativeFiltering(user_ids, movie_ids),
        wcr_factory()
    ]

    n = 30

    for method in methods:
        iterations = 0
        all_hits = 0
        for train_df, test_df in user_leave_on_out(ratings):
            train_ratings = train_df.values[:, :3]
            user_id, item_id, ratings = test_df.values[:, :3][0]
            method.fit(train_ratings)
            pred_ids = method.get_recommendations(user_id, n)
            hits = hit_rate(gt_items_idx=[item_id.astype(int)], predicted_items_idx=pred_ids)

            ###
            # SPACE OF OTHER METRICS
            ###

            all_hits += hits
            iterations += 1

            if hits > 0:
                print(f"{method.__class__}: {all_hits}/{iterations}")


if __name__ == '__main__':
    main()
