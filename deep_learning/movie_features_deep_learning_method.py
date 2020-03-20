from interfaces import RecommendationMethod
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from settings import PATH_TO_PROJECT
import os


class MovieFeaturesDeepLearningMethod(RecommendationMethod):
    def __init__(self):
        self.model = None
        self.user_ratings = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def fit(self, train_user_items_ratings, test_user_items_ratings=None, batch_size=20, epochs=20):
        self.model = self._build_model(train_user_items_ratings[0][0].shape[0])
        eval_test = test_user_items_ratings is not None

        train_ds = self._generate_dataset(train_user_items_ratings, batch_size)

        if eval_test:
            test_ds = self._generate_dataset(test_user_items_ratings, batch_size)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()

        for e in range(epochs):
            for train_data in tqdm(train_ds, total=len(train_user_items_ratings) // batch_size):
                users, items, ratings = train_data
                self.train_step(optimizer, loss_object, users, items, ratings)

            if eval_test:
                for test_data in tqdm(test_ds, total=len(test_user_items_ratings) // batch_size):
                    users, items, ratings = test_data
                    self.test_step(loss_object, users, items, ratings)

                template = 'Epoch {}, Loss: {}, Test Loss: {}'
                print(template.format(e + 1,
                                      self.train_loss.result().numpy(),
                                      self.test_loss.result().numpy()))
            else:
                template = 'Epoch {}, Loss: {}'
                print(template.format(e + 1,
                                      self.train_loss.result().numpy()))

        self.model.save(os.path.join(PATH_TO_PROJECT, 'deep_learning', 'saved_models', 'model.h5'))

    def load_model(self, filepath, input_size):
        self.model = self._build_model(input_size=input_size)
        self.model.load_weights(filepath=filepath)

    def get_recommendations(self, user, movies, k=10):
        user_input = np.repeat(np.expand_dims(user, axis=0), movies.shape[0], axis=0)
        movies_input = movies

        recommendations = self.model.predict([user_input, movies_input]).squeeze()
        recommendations_idx = np.argsort(recommendations)[::-1]
        return recommendations_idx[:k]

    def _generate_dataset(self, data, batch_size):
        users_ids = np.array([r[0] for r in data])
        items_ids = np.array([r[1] for r in data])
        ratings_ids = np.array([r[2] for r in data])
        return tf.data.Dataset.from_tensor_slices((users_ids, items_ids, ratings_ids)).shuffle(10000, seed=56).batch(
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

    def _build_model(self, input_size):
        user_inputs = tf.keras.Input(shape=(input_size,))
        item_inputs = tf.keras.Input(shape=(input_size,))

        user_hidden = tf.keras.layers.Dense(512, activation=tf.nn.relu)(user_inputs)

        item_hidden = tf.keras.layers.Dense(512, activation=tf.nn.relu)(item_inputs)

        concatenated = tf.keras.layers.Concatenate(axis=1)([item_hidden, user_hidden])

        flattened = tf.keras.layers.Flatten()(concatenated)
        dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, trainable=True)(flattened)
        dense_2 = tf.keras.layers.Dense(32, activation=tf.nn.relu, trainable=True)(dense_1)
        dense_3 = tf.keras.layers.Dense(1, activation=tf.nn.relu, trainable=True)(dense_2)

        model = tf.keras.models.Model([item_inputs, user_inputs], dense_3)
        return model
