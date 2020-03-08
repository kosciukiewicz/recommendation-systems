import pandas as pd
import numpy as np

from settings import PATH_TO_DATA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collaborative_filtering.memory_based_collaborative_filtering import MemoryBasedCollaborativeFiltering
from collaborative_filtering.svd_collaborative_filtering import SVDCollaborativeFiltering
from collaborative_filtering.neural_collaborative_filtering import NeuralCollaborativeFiltering
from tqdm import tqdm


def main():
    # loading data
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    user_column = 'userId'
    item_column = 'movieId'
    rating_column = 'rating'

    ratings_samples = user_ratings_df[[user_column, item_column, rating_column]].values
    ratings_samples = [(int(r[0]), int(r[1]), r[2]) for r in ratings_samples]

    train_ratings, test_ratings = train_test_split(ratings_samples, train_size=0.8, shuffle=True, random_state=123)

    # CLASSIC MEMORY BASED
    # method = MemoryBasedCollaborativeFiltering
    #
    # cf = method(users_ids=user_ratings_df[user_column].unique(),
    #             items_ids=user_ratings_df[item_column].unique())
    #
    # cf.fit(user_items_ratings=train_ratings)

    # SVD MATRIX FACTORISATION
    # method = SVDCollaborativeFiltering
    #
    # cf = method(users_ids=user_ratings_df[user_column].unique(),
    #             items_ids=user_ratings_df[item_column].unique())
    #
    # cf.fit(user_items_ratings=train_ratings, n_factors=100)

    # NEURAL
    method = NeuralCollaborativeFiltering

    cf = method(users_ids=user_ratings_df[user_column].unique(),
                items_ids=user_ratings_df[item_column].unique())

    cf.fit(train_user_item_ratings=train_ratings, test_user_item_ratings=test_ratings, n_factors=32, batch_size=50,
           epochs=20)

    y_true = []
    y_pred = []

    for i in tqdm(range(len(test_ratings))):
        user, item, ratings = test_ratings[i]
        predicted_rating = cf.predict(user_id=user, item_id=item)

        y_true.append(ratings)
        y_pred.append(predicted_rating)

    print(mean_squared_error(y_true=y_true, y_pred=y_pred))


if __name__ == '__main__':
    main()
