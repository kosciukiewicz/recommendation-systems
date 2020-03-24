from interfaces import RecommendationMethod
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from settings import PATH_TO_PROJECT
from collaborative_filtering.utils import create_user_items_rating_matrix_w_indexer
import os


class MovieFeaturesDeepLearningMethod(RecommendationMethod):
    def __init__(self, indexer, movies_features, user_features, model_name):
        self.indexer = indexer
        self.name = model_name
        self.movies_features = movies_features
        self.user_features = user_features
        self.model = None
        self.user_ratings = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def fit(self, train_user_items_ratings, test_user_items_ratings=None, batch_size=20, epochs=20, early_stopping=-1):
        self.model = self._build_model()
        eval_test = test_user_items_ratings is not None

        #train_user_items_ratings.extend(self._generate_negative_samples(train_user_items_ratings, 1))

        train_ds = self._generate_dataset(train_user_items_ratings, batch_size)

        if eval_test:
            test_ds = self._generate_dataset(test_user_items_ratings, batch_size)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

        prev_test_loss = float("inf")
        patience = early_stopping
        done_epochs = 0

        while done_epochs < epochs and patience != 0:
            for train_data in tqdm(train_ds, total=len(train_user_items_ratings) // batch_size):
                users, items, ratings = train_data
                self.train_step(optimizer, loss_object, users, items, ratings)

            if eval_test:
                for test_data in tqdm(test_ds, total=len(test_user_items_ratings) // batch_size):
                    users, items, ratings = test_data
                    self.test_step(loss_object, users, items, ratings)

                template = 'Epoch {}, Loss: {}, Test Loss: {}'
                print(template.format(done_epochs + 1,
                                      self.train_loss.result().numpy(),
                                      self.test_loss.result().numpy()))

                if self.test_loss.result().numpy() < prev_test_loss:
                    prev_test_loss = self.test_loss.result().numpy()
                    patience = early_stopping
                else:
                    patience -= 1

            else:
                template = 'Epoch {}, Loss: {}'
                print(template.format(done_epochs + 1,
                                      self.train_loss.result().numpy()))

            done_epochs += 1

        if patience == 0:
            print("Early stopped")

        self.model.save(os.path.join(PATH_TO_PROJECT, 'deep_learning', 'saved_models', f'{self.name}.h5'))

    def load_model(self, filepath):
        self.model = self._build_model()
        self.model.load_weights(filepath=filepath)

    def get_recommendations(self, user_id, k=None):
        user = self.user_features[self.indexer.get_user_internal_id(user_id), :]
        user_input = np.repeat(np.expand_dims(user, axis=0), self.movies_features.shape[0], axis=0)
        movies_input = self.movies_features

        recommendations = self.model.predict([user_input, movies_input]).squeeze()
        recommendations_idx = np.argsort(recommendations)[::-1]
        recommendations_idx = [self.indexer.get_movie_id(internal_id) for internal_id in recommendations_idx]
        return recommendations_idx[:k] if k is not None else recommendations_idx

    def _generate_negative_samples(self, data, count_for_one_user):
        new_data = []
        items_ids = np.arange(0, len(self.indexer.internal_id_to_movie_id_dict.keys()))
        user_ratings = create_user_items_rating_matrix_w_indexer(data, self.indexer)

        for u, i, r in data:
            non_rated_movies = user_ratings[self.indexer.get_user_internal_id(u), :] == 0
            ratings_to_sample = np.random.choice(items_ids[non_rated_movies], count_for_one_user)
            for s in ratings_to_sample:
                new_data.append((u, self.indexer.get_movie_id(s), 0))

        return new_data

    def _generate_dataset(self, data, batch_size):
        users = np.array([self.user_features[self.indexer.get_user_internal_id(r[0])] for r in data])
        items = np.array([self.movies_features[self.indexer.get_movie_internal_id(r[1])] for r in data])
        ratings = np.array([r[2] for r in data])
        return tf.data.Dataset.from_tensor_slices((users, items, ratings)).shuffle(10000, seed=56).batch(
            batch_size)

    @tf.function
    def test_step(self, loss_object, user_input, item_input, labels):
        predictions = self.model([user_input, item_input])
        t_loss = loss_object(labels, predictions)

        self.test_loss(t_loss)

    @tf.function
    def train_step(self, optimizer, loss_object, user_input, item_input, labels):
        with tf.GradientTape() as tape:
            predictions = self.model([user_input, item_input], training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def _build_model(self):
        user_inputs = tf.keras.Input(shape=(self.movies_features.shape[1],))
        item_inputs = tf.keras.Input(shape=(self.movies_features.shape[1],))

        user_hidden = tf.keras.layers.Dense(256, activation=tf.nn.tanh)(user_inputs)

        item_hidden = tf.keras.layers.Dense(256, activation=tf.nn.tanh)(item_inputs)

        concatenated = tf.keras.layers.Concatenate(axis=1)([item_hidden, user_hidden])

        flattened = tf.keras.layers.Flatten()(concatenated)
        dense_1 = tf.keras.layers.Dense(128, activation=tf.nn.tanh, trainable=True)(flattened)
        dense_2 = tf.keras.layers.Dense(16, activation=tf.nn.tanh, trainable=True)(dense_1)
        dropout = tf.keras.layers.Dropout(0.5)(dense_2)
        dense_3 = tf.keras.layers.Dense(1, activation=tf.nn.relu, trainable=True)(dropout)

        model = tf.keras.models.Model([item_inputs, user_inputs], dense_3)
        return model
