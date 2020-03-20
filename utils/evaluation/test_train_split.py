import random
import numpy as np
import pandas as pd

user_column = 'userId'
item_column = 'movieId'
rating_column = 'rating'


def user_leave_on_out(user_ratings_df, timestamp_column=None, make_user_folds=True):
    """
    Make leave one out folds for each user
    :param user_ratings_df: default user items ratings pandas dataframe
    :param timestamp_column: timestamp column name in user_ratings_df for setting last reting as test rating
     If None setting random user rating as test rating
    :param make_user_folds: if True makes fold generator for each user leaving last or random rating as test rating.
     If False returning one train dataset and one test set with oe rating for each user
    :return:
    """

    user_ids = user_ratings_df[user_column].unique()
    test_indices = []
    for user_id in user_ids:
        user_ratings = user_ratings_df[user_ratings_df[user_column] == user_id]

        if timestamp_column is not None:
            user_ratings = user_ratings.sort_values(by=timestamp_column)
            test_index = len(user_ratings) - 1
        else:
            test_index = random.randint(0, len(user_ratings) - 1)

        indices = user_ratings.index.values
        test_indices.append(indices[test_index])

    if make_user_folds:
        for i in test_indices:
            yield user_ratings_df.loc[~user_ratings_df.index.isin([i])], user_ratings_df.loc[[i]]
    else:
        yield user_ratings_df.loc[~user_ratings_df.index.isin(test_indices)], user_ratings_df.loc[test_indices]
