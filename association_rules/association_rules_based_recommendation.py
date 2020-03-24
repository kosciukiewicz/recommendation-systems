import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from interfaces import RecommendationMethod
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


class AssociationRulesRecommendation(RecommendationMethod):

    def __init__(self, indexer, min_support):
        self.indexer = indexer
        self.min_support = min_support

        self.frequent_itemsets = None

    def fit(self, ratings):
        ratings = self.map_ratings_to_internal_representation(ratings)
        dataset = self.get_dataset(ratings)
        self.frequent_itemsets = self.calc_frequent_itemset(dataset)

    def get_recommendations(self, user_id, n):
        pass

    def map_ratings_to_internal_representation(self, ratings):
        return [(self.indexer.get_user_internal_id(user_id),
                 self.indexer.get_movie_internal_id(movie_id), rating) for
                user_id, movie_id, rating in ratings]

    def get_dataset(self, ratings):
        transactions = [[movie_id for _, movie_id, _ in movies_ids]
                        for user_id, movies_ids in
                        groupby(ratings, key=itemgetter(0))]
        transaction_encoder = TransactionEncoder()
        one_hot = transaction_encoder.fit(transactions).transform(
            transactions)
        return pd.DataFrame(one_hot, columns=transaction_encoder.columns_)

    def calc_frequent_itemset(self, dataset):
        frequent_itemsets = apriori(dataset,
                                    min_support=self.min_support,
                                    use_colnames=True)
        frequent_itemsets = frequent_itemsets[
            frequent_itemsets['itemsets'].apply(
                lambda itemset: len(itemset)) > 1]
        return frequent_itemsets
