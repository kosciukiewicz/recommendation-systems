import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tqdm import tqdm

MLP_DENSE_LAYERS_SIZE = [32, 16, 4]


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
        self.user_ratings = None
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

    def _build_gmf_model(self, n_factors):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        mf_user_embedding = tf.keras.layers.Embedding(self.num_users, n_factors, input_length=1,
                                                      name="gmf_user_embedding")(user_inputs)
        mf_user_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mf_user_embedding)

        mf_item_embedding = tf.keras.layers.Embedding(self.num_items, n_factors, input_length=1,
                                                      name="gmf_item_embedding")(item_inputs)
        mf_item_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mf_item_embedding)

        gmf_dot_product = tf.keras.layers.Dot(axes=1)([mf_user_embedding, mf_item_embedding])
        gmf_dot_product = tf.keras.layers.Flatten()(gmf_dot_product)

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=False, name="gmf_dense_output")(
            gmf_dot_product)

        model = tf.keras.models.Model([user_inputs, item_inputs], final_dense)

        return model

    def _build_mlp_model(self, n_factors):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        mlp_user_embedding = tf.keras.layers.Embedding(self.num_users, n_factors, input_length=1,
                                                       name="mlp_user_embedding")(user_inputs)
        mlp_user_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mlp_user_embedding)

        mlp_item_embedding = tf.keras.layers.Embedding(self.num_items, n_factors, input_length=1,
                                                       name="mlp_item_embedding")(item_inputs)
        mlp_item_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mlp_item_embedding)

        mlp_concatenation = tf.keras.layers.Concatenate(axis=1)(
            [mlp_user_embedding, mlp_item_embedding])
        mlp_concatenation = tf.keras.layers.Flatten()(mlp_concatenation)

        dense_output = mlp_concatenation
        for layer_size in MLP_DENSE_LAYERS_SIZE:
            dense_output = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, trainable=False,
                                                 name=f"mlp_dense_{layer_size}")(dense_output)

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=False, name="mlp_dense_output")(
            dense_output)

        model = tf.keras.models.Model([user_inputs, item_inputs], final_dense)

        return model

    def _build_neu_mf_model(self, n_factors, pretrained_gmf_model, pretrained_mlp_model):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        mf_user_embedding = tf.keras.layers.Embedding(self.num_users, n_factors, input_length=1,
                                                      name="gmf_user_embedding",
                                                      weights=pretrained_gmf_model.get_layer(
                                                          "gmf_user_embedding").get_weights())(user_inputs)
        mf_user_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mf_user_embedding)

        mf_item_embedding = tf.keras.layers.Embedding(self.num_items, n_factors, input_length=1,
                                                      name="gmf_item_embedding",
                                                      weights=pretrained_gmf_model.get_layer(
                                                          "gmf_item_embedding").get_weights())(item_inputs)
        mf_item_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mf_item_embedding)

        mlp_user_embedding = tf.keras.layers.Embedding(self.num_users, n_factors, input_length=1,
                                                       name="mlp_user_embedding",
                                                       weights=pretrained_mlp_model.get_layer(
                                                           "mlp_user_embedding").get_weights())(user_inputs)
        mlp_user_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mlp_user_embedding)

        mlp_item_embedding = tf.keras.layers.Embedding(self.num_items, n_factors, input_length=1,
                                                       name="mlp_item_embedding",
                                                       weights=pretrained_mlp_model.get_layer(
                                                           "mlp_item_embedding").get_weights())(item_inputs)
        mlp_item_embedding = tf.keras.layers.Reshape(target_shape=(n_factors,))(mlp_item_embedding)

        gmf_dot_product = tf.keras.layers.Dot(axes=1)([mf_user_embedding, mf_item_embedding])
        gmf_dot_product = tf.keras.layers.Flatten()(gmf_dot_product)

        mlp_concatenation = tf.keras.layers.Concatenate(axis=1)(
            [mlp_user_embedding, mlp_item_embedding])
        mlp_concatenation = tf.keras.layers.Flatten()(mlp_concatenation)

        dense_output = mlp_concatenation

        for layer_size in MLP_DENSE_LAYERS_SIZE:
            dense_output = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, trainable=False,
                                                 name=f"mlp_dense_{layer_size}",
                                                 weights=pretrained_mlp_model.get_layer(
                                                     f"mlp_dense_{layer_size}").get_weights())(dense_output)

        neu_concat = tf.keras.layers.Concatenate(axis=1)([gmf_dot_product, dense_output])

        final_gmf_weights = pretrained_gmf_model.get_layer("gmf_dense_output").get_weights()
        final_mlp_wights = pretrained_mlp_model.get_layer("mlp_dense_output").get_weights()

        new_final_weights = np.concatenate([
            final_gmf_weights[0],
            final_mlp_wights[0],
        ], axis=0)

        new_final_biases = final_gmf_weights[1] + final_mlp_wights[1]

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=False,
                                            weights=[0.5 * new_final_weights, 0.5 * new_final_biases])(neu_concat)

        model = tf.keras.models.Model([user_inputs, item_inputs], final_dense)

        return model

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

    def _generate_dataset(self, data, batch_size):
        users_ids = np.expand_dims(np.array([self.user_id_to_id_vocab[r[0]] for r in data]), axis=1)
        items_ids = np.expand_dims(np.array([self.item_id_to_id_vocab[r[1]] for r in data]), axis=1)
        ratings_ids = np.expand_dims(np.array([r[2] for r in data]), axis=1)
        return tf.data.Dataset.from_tensor_slices((users_ids, items_ids, ratings_ids)).shuffle(10000, seed=56).batch(
            batch_size)

    @tf.function
    def gmf_pretrain_step(self, model, optimizer, loss_object, user_input, item_input, labels):
        with tf.GradientTape() as tape:
            predictions = model([user_input, item_input], training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)

    @tf.function
    def mlp_pretrain_step(self, model, optimizer, loss_object, user_input, item_input, labels):
        with tf.GradientTape() as tape:
            predictions = model([user_input, item_input], training=True)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)

    def _pretrain_models(self, train_user_item_ratings, epochs=20, batch_size=100,
                         n_factors=None):

        print("Starting pretraining phase...")

        gmf_model = self._build_gmf_model(n_factors=n_factors)
        mlp_model = self._build_mlp_model(n_factors=n_factors)

        train_ds = self._generate_dataset(train_user_item_ratings, batch_size)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()

        for e in range(epochs):
            for train_data in tqdm(train_ds, total=len(train_user_item_ratings) // batch_size):
                user_id, item_id, rating = train_data
                self.gmf_pretrain_step(gmf_model, optimizer, loss_object, user_id, item_id, rating)

            template = 'Epoch {}, Loss: {}'

            print(template.format(e + 1,
                                  self.train_loss.result().numpy()))

        self.train_loss.reset_states()

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()

        for e in range(epochs):
            for train_data in tqdm(train_ds, total=len(train_user_item_ratings) // batch_size):
                user_id, item_id, rating = train_data
                self.mlp_pretrain_step(mlp_model, optimizer, loss_object, user_id, item_id, rating)

            template = 'Epoch {}, Loss: {}'

            print(template.format(e + 1,
                                  self.train_loss.result().numpy()))

        self.train_loss.reset_states()

        return gmf_model, mlp_model

    def fit(self, train_user_item_ratings, test_user_item_ratings, epochs=10, batch_size=100, n_factors=16):
        self.user_ratings = np.zeros((len(self.id_to_user_id_vocab.keys()), len(self.id_to_item_id_vocab.keys())))

        for i in tqdm(range(len(train_user_item_ratings))):
            user, item, rating = train_user_item_ratings[i]
            user_index = self.user_id_to_id_vocab[int(user)]
            item_index = self.item_id_to_id_vocab[int(item)]

            self.user_ratings[user_index, item_index] = rating

        gmf_model, mlp_model = self._pretrain_models(train_user_item_ratings, epochs=1, n_factors=n_factors)

        self.model = self._build_neu_mf_model(n_factors, pretrained_gmf_model=gmf_model, pretrained_mlp_model=mlp_model)

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()

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
        user_input = np.array([self.user_id_to_id_vocab[user_id]])
        item_input = np.array([self.item_id_to_id_vocab[item_id]])
        return self.model.predict([user_input, item_input]).squeeze().tolist()

    def get_top(self, user_id, k=10):
        user_input = np.full((len(self.item_id_to_id_vocab.keys()), 1), fill_value=self.user_id_to_id_vocab[user_id])
        item_input = np.expand_dims(np.arange(0, len(self.item_id_to_id_vocab.keys())), axis=1)

        non_rated_user_movies = self.user_ratings[self.user_id_to_id_vocab[user_id], :] == 0
        recommendations = self.model.predict([user_input, item_input]).squeeze()
        recommendations_idx = np.argsort(recommendations)[::-1]
        return [self.id_to_item_id_vocab[i] for i in recommendations_idx[:k] if non_rated_user_movies[i]]
