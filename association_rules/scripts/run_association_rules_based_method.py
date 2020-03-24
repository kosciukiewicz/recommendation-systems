from association_rules.association_rules_based_recommendation import \
    AssociationRulesRecommendation
import pandas as pd
import os
import numpy as np
from settings import PATH_TO_DATA
from utils.id_indexer import Indexer
from utils.features_extraction.movie_lens_features_extractor import \
    FeaturesExtractor

pd.set_option('display.max_rows', 30000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1)


def save_extracted_features(dataset_path):
    features_extractor = FeaturesExtractor(dataset_path)
    features_extractor.save()


def replace_existing_movies_ids_with_titles(ids, movies_titles):
    def try_replace_id_with_title(id):
        title = movies_titles.get(id)
        return title if title else int(id)

    return {id: try_replace_id_with_title(id) for id in ids}


def get_frequent_itemset_with_titles(arbr, indexer, dataset_path):
    features_extractor = FeaturesExtractor(dataset_path)
    movies_data = features_extractor.load()
    movies_titles = dict(zip(movies_data['id'], movies_data['title']))

    movies_titles = replace_existing_movies_ids_with_titles(
        indexer.movie_id_to_internal_id_dict.keys(), movies_titles)
    frequent_itemsets = arbr.frequent_itemsets['itemsets'].apply(
        lambda itemset: [movies_titles[indexer.internal_id_to_movie_id_dict[id]] for id in itemset])
    return frequent_itemsets


def find_min_confidence(ratings, indexer, min_support):
    for min_confidence in [0.6, 0.4, 0.3, 0.2, 0.15, 0.1]:
        arbr = AssociationRulesRecommendation(indexer, min_support, min_confidence)
        arbr.fit(ratings)
        print(arbr.rules.shape)


def main():
    dataset_path = os.path.join(PATH_TO_DATA, 'raw/the-movies-dataset')
    ratings = pd.read_csv(
        os.path.join(dataset_path, 'ratings_small.csv')).values[:, :-1]

    indexer = Indexer(user_ids=np.unique(ratings[:, 0]),
                      movies_ids=np.unique(ratings[:, 1]))

    arbr = AssociationRulesRecommendation(indexer, .15, .2)
    arbr.fit(ratings)

    print(arbr.get_recommendations(1, 5))

    # print(len(set(ratings[:, 1])))




if __name__ == '__main__':
    main()
