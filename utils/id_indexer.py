def _generate_mapping(ids):
    id_to_data_id_vocab = {}

    id = 0
    for i in ids:
        if i not in id_to_data_id_vocab.values():
            id_to_data_id_vocab[id] = i
            id += 1

    return id_to_data_id_vocab, {v: k for k, v in id_to_data_id_vocab.items()}


class Indexer:
    def __init__(self, user_ids=None, movies_ids=None):
        if user_ids is not None:
            self.internal_id_to_user_id_dict, self.user_id_to_internal_id_dict = _generate_mapping(ids=user_ids)
        else:
            self.internal_id_to_user_id_dict = None
            self.user_id_to_internal_id_dict = None

        if movies_ids is not None:
            self.internal_id_to_movie_id_dict, self.movie_id_to_internal_id_dict = _generate_mapping(
                ids=movies_ids)
        else:
            self.internal_id_to_movie_id_dict = None
            self.movie_id_to_internal_id_dict = None

    def set_user_ids(self, user_ids):
        self.internal_id_to_user_id_dict, self.user_id_to_internal_id_dict = _generate_mapping(ids=user_ids)

    def set_movies_id(self, movies_ids):
        self.internal_id_to_movie_id_dict, self.movie_id_to_internal_id_dict = _generate_mapping(
            ids=movies_ids)

    def get_user_id(self, internal_id):
        return self.internal_id_to_user_id_dict[internal_id]

    def get_user_internal_id(self, user_id):
        return self.user_id_to_internal_id_dict[user_id]

    def get_movie_id(self, internal_id):
        return self.internal_id_to_movie_id_dict[internal_id]

    def get_movie_internal_id(self, movie_id):
        return self.movie_id_to_internal_id_dict[movie_id]