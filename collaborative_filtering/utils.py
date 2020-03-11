from tqdm import tqdm
import numpy as np


def create_id_vocab(data):
    id_to_data_id_vocab = {}

    id = 0
    for i in data:
        if i not in id_to_data_id_vocab.values():
            id_to_data_id_vocab[id] = i
            id += 1

    return id_to_data_id_vocab, {v: k for k, v in id_to_data_id_vocab.items()}


def create_user_items_rating_matrix(data, user_id_to_id_mapping, item_id_to_id_mapping):
    user_ratings = np.zeros((len(user_id_to_id_mapping.values()), len(item_id_to_id_mapping.values())))

    for i in tqdm(range(len(data))):
        user, item, rating = data[i]
        user_index = user_id_to_id_mapping[int(user)]
        item_index = item_id_to_id_mapping[int(item)]

        user_ratings[user_index, item_index] = rating

    return user_ratings