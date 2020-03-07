import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tqdm import tqdm


def create_id_vocab(data):
    id_to_data_id_vocab = {}

    id = 0
    for i in data:
        if i not in id_to_data_id_vocab.values():
            id_to_data_id_vocab[id] = i
            id += 1

    return id_to_data_id_vocab, {v: k for k, v in id_to_data_id_vocab.items()}


class NeuralCollaborativeFiltering:
    def __init__(self, users_ids, items_ids):
        self.id_to_user_id_vocab, self.user_id_to_id_vocab = create_id_vocab(users_ids)
        self.id_to_item_id_vocab, self.item_id_to_id_vocab = create_id_vocab(items_ids)
        self.model = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    @property
    def num_users(self):
        return len(self.id_to_user_id_vocab.keys())

    @property
    def num_items(self):
        return len(self.id_to_item_id_vocab.keys())

    def _build_simple_dot_model(self, n_factors):
        users_embedding = tf.keras.Sequential()
        users_embedding.add(tf.keras.layers.Embedding(self.num_users, n_factors, input_length=1))
        users_embedding.add(tf.keras.layers.Reshape(target_shape=(n_factors,)))
        items_embedding = tf.keras.models.Sequential()
        items_embedding.add(tf.keras.layers.Embedding(self.num_items, n_factors, input_length=1))
        items_embedding.add(tf.keras.layers.Reshape(target_shape=(n_factors,)))
        dot = tf.keras.layers.Dot(axes=1)([users_embedding.output, items_embedding.output])
        model = tf.keras.models.Model([users_embedding.input, items_embedding.input], dot)

        return model

    def _build_concat_model(self, n_factors):
        users_embedding = tf.keras.Sequential()
        users_embedding.add(tf.keras.layers.Embedding(self.num_users, n_factors, input_length=1))
        users_embedding.add(tf.keras.layers.Reshape(target_shape=(n_factors,)))
        items_embedding = tf.keras.models.Sequential()
        items_embedding.add(tf.keras.layers.Embedding(self.num_items, n_factors, input_length=1))
        items_embedding.add(tf.keras.layers.Reshape(target_shape=(n_factors,)))
        concatenated = tf.keras.layers.Concatenate(axis=1)([users_embedding.output, items_embedding.output])
        flattened = tf.keras.layers.Flatten()(concatenated)
        dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, trainable=False)(flattened)
        dense_2 = tf.keras.layers.Dense(32, activation=tf.nn.relu, trainable=False)(dense_1)
        dense_3 = tf.keras.layers.Dense(1, activation=tf.nn.relu, trainable=False)(dense_2)

        model = tf.keras.models.Model([users_embedding.input, items_embedding.input], dense_3)

        return model

    @tf.function
    def test_step(self, loss_object, user_input, item_input, labels):
        predictions = self.model([user_input, item_input])
        t_loss = loss_object(labels, predictions)

        self.test_loss(t_loss)

    def train_step(self, optimizer, loss_object, user_input, item_input, labels):
        with tf.GradientTape() as tape:
            predictions = self.model([user_input, item_input], training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)

    def _generate_dataset(self, data, batch_size):
        users_ids = np.expand_dims(np.array([self.user_id_to_id_vocab[r[0]] for r in data]), axis=1)
        items_ids = np.expand_dims(np.array([self.item_id_to_id_vocab[r[1]] for r in data]), axis=1)
        ratings_ids = np.expand_dims(np.array([r[2] for r in data]), axis=1)
        return tf.data.Dataset.from_tensor_slices((users_ids, items_ids, ratings_ids)).shuffle(10000, seed=56).batch(
            batch_size)

    def fit(self, train_user_item_ratings, test_user_item_ratings, epochs=10, batch_size=100, n_factors=None):
        self.model = self._build_concat_model(n_factors)
        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adamax()

        train_ds = self._generate_dataset(train_user_item_ratings, batch_size)
        test_ds = self._generate_dataset(test_user_item_ratings, batch_size)

        for e in range(epochs):
            for train_data in tqdm(train_ds, total=len(train_user_item_ratings) // batch_size):
                user_id, item_id, rating = train_data
                self.train_step(optimizer, loss_object, user_id, item_id, rating)

            for test_data in tqdm(test_ds, total=len(test_user_item_ratings) // batch_size):
                user_id, item_id, rating = test_data
                self.test_step(loss_object, user_id, item_id, rating)

            template = 'Epoch {}, Loss: {}, Test Loss: {}'

            print(template.format(e + 1,
                                  self.train_loss.result().numpy(),
                                  self.test_loss.result().numpy()))

    def predict(self, user_id, item_id):
        pass
