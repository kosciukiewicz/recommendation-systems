import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from interfaces import RecommendationMethod
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from utils.matrix_generator import RatingsBasedMatrixGenerator


class AssociationRulesRecommendation(RecommendationMethod):

    def __init__(self, indexer, min_support, min_confidence):
        self.indexer = indexer
        self.min_support = min_support
        self.min_confidence = min_confidence

        self.frequent_itemsets = None
        self.rules = None
        self.users_ratings_matrix = None

    def fit(self, ratings):
        ratings = self.map_ratings_to_internal_representation(ratings)
        dataset = self.get_dataset(ratings)
        self.frequent_itemsets = self.calc_frequent_itemset(dataset)
        self.rules = self.calc_association_rules()
        self.users_ratings_matrix = RatingsBasedMatrixGenerator(
            self.indexer).generate_user_matrix(ratings)

    def get_recommendations(self, user_id, n):
        user_id = self.indexer.get_user_internal_id(user_id)
        sorted_rule_ids = self.sort_rules_ids_for_user(user_id)
        watched_movies = np.nonzero(self.users_ratings_matrix[user_id, :])[0]

        selected_movies = []
        for id in sorted_rule_ids:
            consequents = self.select_consequents(id, selected_movies,
                                                  watched_movies)
            if len(selected_movies) + len(consequents) >= n:
                return self.map_final_recommendation(
                    selected_movies + consequents[:n - len(selected_movies)])
            else:
                selected_movies += consequents

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
        # frequent_itemsets = frequent_itemsets[
        #     frequent_itemsets['itemsets'].apply(
        #         lambda itemset: len(itemset)) > 1]
        return frequent_itemsets

    def calc_association_rules(self):
        return association_rules(self.frequent_itemsets,
                                 metric="confidence",
                                 min_threshold=self.min_confidence)

    def sort_rules_ids_for_user(self, user_id):
        def score_rule(row):
            return np.mean(
                [self.users_ratings_matrix[user_id, movie_id]
                 for movie_id in row['antecedents']]
            ) * row['lift']

        return np.argsort(-self.rules.apply(score_rule, axis=1).values)

    def select_consequents(self, rule_id, selected_movies, watched_movies):
        all_consequents = self.rules.iloc[rule_id]['consequents']
        return [movie_id for movie_id in all_consequents if
                movie_id not in selected_movies and
                movie_id not in watched_movies]

    def map_final_recommendation(self, intarnal_movies_ids):
        return [self.indexer.get_movie_id(internal_id)
                for internal_id in intarnal_movies_ids]
