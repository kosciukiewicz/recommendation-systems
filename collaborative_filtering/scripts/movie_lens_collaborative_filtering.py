import pandas as pd
import numpy as np

from settings import PATH_TO_DATA, CHECKPOINTS_DIRECTORY
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collaborative_filtering.memory_based_collaborative_filtering import MemoryBasedCollaborativeFiltering
from collaborative_filtering.svd_collaborative_filtering import SVDCollaborativeFiltering
from collaborative_filtering.neural_collaborative_filtering import NeuralCollaborativeFiltering
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def get_movie_id_to_feature_mapping(movies_metadata_df):
    mapping = {}
    for i, row in movies_metadata_df.iterrows():
        features = {
            "title": row["title"],
            "id": row["id"],
        }

        mapping[int(row['id'])] = features

    return mapping


def main():
    # loading data
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    movies_metadata = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/movies_metadata_clean.csv")
    movie_id_features_dict = get_movie_id_to_feature_mapping(movies_metadata)

    user_column = 'userId'
    item_column = 'movieId'
    rating_column = 'rating'

    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]
    ratings_samples = user_ratings_df[[user_column, item_column, rating_column]].values
    ratings_samples = [(int(r[0]), int(r[1]), r[2]) for r in ratings_samples]

    train_ratings, test_ratings = train_test_split(ratings_samples, train_size=0.8, shuffle=True, random_state=123)

    sample_user_rated_movies = \
        user_ratings_df[user_ratings_df[user_column] == 1].sort_values(rating_column, ascending=False)[
            [item_column]].values.squeeze()

    # CLASSIC MEMORY BASED
    #method = MemoryBasedCollaborativeFiltering
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
    # cf.fit(user_items_ratings=train_ratings, n_factors=670)


    # NEURAL
    method = NeuralCollaborativeFiltering

    cf = method(users_ids=user_ratings_df[user_column].unique(),
                items_ids=user_ratings_df[item_column].unique())

    cf.load_model(f"{CHECKPOINTS_DIRECTORY}/model.h5", train_ratings, n_factors=32)

    # cf.fit(train_user_item_ratings=train_ratings, test_user_item_ratings=test_ratings, n_factors=32, batch_size=20,
    #        epochs=20)

    recommended_items = cf.get_recommendations(k=10, user_id=1)

    print([movie_id_features_dict[i]['title'] for i in sample_user_rated_movies])
    print([movie_id_features_dict[i]['title'] for i in recommended_items])


def load_data():
    user_ratings_df = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/ratings_small.csv")
    movies_metadata = pd.read_csv(f"{PATH_TO_DATA}/raw/the-movies-dataset/movies_metadata_clean.csv")
    movie_id_features_dict = get_movie_id_to_feature_mapping(movies_metadata)

    user_column = 'userId'
    item_column = 'movieId'
    rating_column = 'rating'

    user_ratings_df = user_ratings_df[user_ratings_df[item_column].isin(movie_id_features_dict.keys())]
    ratings_samples = user_ratings_df[[user_column, item_column, rating_column]].values
    ratings_samples = [(int(r[0]), int(r[1]), r[2]) for r in ratings_samples]
    return ratings_samples


def count_mse(method, test_data):
    y_true = []
    y_pred = []

    for i in tqdm(range(len(test_data))):
        user, item, ratings = test_data[i]
        predicted_rating = method.predict(user_id=user, item_id=item)

        y_true.append(ratings)
        y_pred.append(predicted_rating)

    scaler = MinMaxScaler()
    scaler.fit(np.array([y_true, y_pred]).transpose())
    data = scaler.transform(np.array([y_true, y_pred]).transpose())
    print(mean_squared_error(y_true=data[:, 0], y_pred=data[:, 1]))

if __name__ == '__main__':
    main()
