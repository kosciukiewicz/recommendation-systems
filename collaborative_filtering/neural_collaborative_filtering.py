import tensorflow as tf
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from settings import CHECKPOINTS_DIRECTORY
from interfaces import RecommendationMethod
from collaborative_filtering.utils import create_id_vocab, create_user_items_rating_matrix_w_indexer

MLP_DENSE_LAYERS_SIZE = [32, 16, 4]


class NeuralCollaborativeFiltering(RecommendationMethod):
    def __init__(self, indexer, n_factors, model_name):
        self.model = None
        self.name = model_name
        self.indexer = indexer
        self.n_factors = n_factors
        self.user_ratings = None
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    @property
    def num_users(self):
        return len(self.indexer.internal_id_to_user_id_dict.keys())

    @property
    def num_items(self):
        return len(self.indexer.internal_id_to_movie_id_dict.keys())

    def load_model(self, filepath, train_user_item_ratings):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        self.user_ratings = create_user_items_rating_matrix_w_indexer(train_user_item_ratings, self.indexer)

        mf_user_embedding = tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1,
                                                      name="gmf_user_embedding")(user_inputs)
        mf_user_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mf_user_embedding)

        mf_item_embedding = tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1,
                                                      name="gmf_item_embedding")(item_inputs)
        mf_item_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mf_item_embedding)

        mlp_user_embedding = tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1,
                                                       name="mlp_user_embedding")(user_inputs)
        mlp_user_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mlp_user_embedding)

        mlp_item_embedding = tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1,
                                                       name="mlp_item_embedding")(item_inputs)

        mlp_item_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mlp_item_embedding)

        gmf_dot_product = tf.keras.layers.Dot(axes=1)([mf_user_embedding, mf_item_embedding])
        gmf_dot_product = tf.keras.layers.Flatten()(gmf_dot_product)

        mlp_concatenation = tf.keras.layers.Concatenate(axis=1)(
            [mlp_user_embedding, mlp_item_embedding])
        mlp_concatenation = tf.keras.layers.Flatten()(mlp_concatenation)

        dense_output = mlp_concatenation

        for layer_size in MLP_DENSE_LAYERS_SIZE:
            dense_output = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, trainable=True,
                                                 name=f"mlp_dense_{layer_size}")(dense_output)

        neu_concat = tf.keras.layers.Concatenate(axis=1)([gmf_dot_product, dense_output])

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=True)(neu_concat)

        self.model = tf.keras.models.Model([user_inputs, item_inputs], final_dense)
        self.model.load_weights(filepath=filepath)

    def _build_simple_dot_model(self):
        users_embedding = tf.keras.Sequential()
        users_embedding.add(tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1))
        users_embedding.add(tf.keras.layers.Reshape(target_shape=(self.n_factors,)))
        items_embedding = tf.keras.models.Sequential()
        items_embedding.add(tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1))
        items_embedding.add(tf.keras.layers.Reshape(target_shape=(self.n_factors,)))
        dot = tf.keras.layers.Dot(axes=1)([users_embedding.output, items_embedding.output])
        model = tf.keras.models.Model([users_embedding.input, items_embedding.input], dot)

        return model

    def _build_concat_model(self):
        users_embedding = tf.keras.Sequential()
        users_embedding.add(tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1))
        users_embedding.add(tf.keras.layers.Reshape(target_shape=(self.n_factors,)))
        items_embedding = tf.keras.models.Sequential()
        items_embedding.add(tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1))
        items_embedding.add(tf.keras.layers.Reshape(target_shape=(self.n_factors,)))
        concatenated = tf.keras.layers.Concatenate(axis=1)([users_embedding.output, items_embedding.output])
        flattened = tf.keras.layers.Flatten()(concatenated)
        dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, trainable=True)(flattened)
        dense_2 = tf.keras.layers.Dense(32, activation=tf.nn.relu, trainable=True)(dense_1)
        dense_3 = tf.keras.layers.Dense(1, activation=tf.nn.relu, trainable=True)(dense_2)

        model = tf.keras.models.Model([users_embedding.input, items_embedding.input], dense_3)

        return model

    def _build_gmf_model(self):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        mf_user_embedding = tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1,
                                                      name="gmf_user_embedding")(user_inputs)
        mf_user_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mf_user_embedding)

        mf_item_embedding = tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1,
                                                      name="gmf_item_embedding")(item_inputs)
        mf_item_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mf_item_embedding)

        gmf_dot_product = tf.keras.layers.Dot(axes=1)([mf_user_embedding, mf_item_embedding])
        gmf_dot_product = tf.keras.layers.Flatten()(gmf_dot_product)

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=True,
                                            name="gmf_dense_output")(
            gmf_dot_product)

        model = tf.keras.models.Model([user_inputs, item_inputs], final_dense)

        return model

    def _build_mlp_model(self):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        mlp_user_embedding = tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1,
                                                       name="mlp_user_embedding")(user_inputs)
        mlp_user_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mlp_user_embedding)

        mlp_item_embedding = tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1,
                                                       name="mlp_item_embedding")(item_inputs)
        mlp_item_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mlp_item_embedding)

        mlp_concatenation = tf.keras.layers.Concatenate(axis=1)(
            [mlp_user_embedding, mlp_item_embedding])
        mlp_concatenation = tf.keras.layers.Flatten()(mlp_concatenation)

        dense_output = mlp_concatenation
        for layer_size in MLP_DENSE_LAYERS_SIZE:
            dense_output = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, trainable=True,
                                                 name=f"mlp_dense_{layer_size}")(dense_output)

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=True, name="mlp_dense_output")(
            dense_output)

        model = tf.keras.models.Model([user_inputs, item_inputs], final_dense)

        return model

    def _build_neu_mf_model(self, pretrained_gmf_model, pretrained_mlp_model):
        user_inputs = tf.keras.Input(shape=(1,))
        item_inputs = tf.keras.Input(shape=(1,))

        mf_user_embedding = tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1,
                                                      name="gmf_user_embedding",
                                                      weights=pretrained_gmf_model.get_layer(
                                                          "gmf_user_embedding").get_weights())(user_inputs)
        mf_user_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mf_user_embedding)

        mf_item_embedding = tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1,
                                                      name="gmf_item_embedding",
                                                      weights=pretrained_gmf_model.get_layer(
                                                          "gmf_item_embedding").get_weights())(item_inputs)
        mf_item_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mf_item_embedding)

        mlp_user_embedding = tf.keras.layers.Embedding(self.num_users, self.n_factors, input_length=1,
                                                       name="mlp_user_embedding",
                                                       weights=pretrained_mlp_model.get_layer(
                                                           "mlp_user_embedding").get_weights())(user_inputs)
        mlp_user_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mlp_user_embedding)

        mlp_item_embedding = tf.keras.layers.Embedding(self.num_items, self.n_factors, input_length=1,
                                                       name="mlp_item_embedding",
                                                       weights=pretrained_mlp_model.get_layer(
                                                           "mlp_item_embedding").get_weights())(item_inputs)
        mlp_item_embedding = tf.keras.layers.Reshape(target_shape=(self.n_factors,))(mlp_item_embedding)

        gmf_dot_product = tf.keras.layers.Dot(axes=1)([mf_user_embedding, mf_item_embedding])
        gmf_dot_product = tf.keras.layers.Flatten()(gmf_dot_product)

        mlp_concatenation = tf.keras.layers.Concatenate(axis=1)(
            [mlp_user_embedding, mlp_item_embedding])
        mlp_concatenation = tf.keras.layers.Flatten()(mlp_concatenation)

        dense_output = mlp_concatenation

        for layer_size in MLP_DENSE_LAYERS_SIZE:
            dense_output = tf.keras.layers.Dense(layer_size, activation=tf.nn.relu, trainable=True,
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

        final_dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, trainable=True,
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

    def _generate_negative_samples(self, data, count_for_one_user):
        new_data = []
        items_ids = np.arange(0, self.user_ratings.shape[1])

        for u, i, r in data:
            non_rated_movies = self.user_ratings[self.indexer.get_user_internal_id(u), :] == 0
            ratings_to_sample = np.random.choice(items_ids[non_rated_movies], count_for_one_user)
            for s in ratings_to_sample:
                new_data.append((u, self.indexer.get_movie_id(s), 0))

        return new_data

    def _generate_dataset(self, data, batch_size):
        users_ids = np.expand_dims(np.array([self.indexer.get_user_internal_id(r[0]) for r in data]), axis=1)
        items_ids = np.expand_dims(np.array([self.indexer.get_movie_internal_id(r[1]) for r in data]), axis=1)
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

    def _pretrain_models(self, train_user_item_ratings, epochs=20, batch_size=100, error_delta=-1.0):

        print("Starting pretraining phase...")

        gmf_model = self._build_gmf_model()
        mlp_model = self._build_mlp_model()

        train_ds = self._generate_dataset(train_user_item_ratings, batch_size)

        loss_object = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

        done_epochs = 0
        prev_loss = float('inf')
        delta = float('inf')

        while done_epochs < epochs and (error_delta == -1.0 or delta > error_delta):
            for train_data in tqdm(train_ds, total=len(train_user_item_ratings) // batch_size):
                user_id, item_id, rating = train_data
                self.gmf_pretrain_step(gmf_model, optimizer, loss_object, user_id, item_id, rating)

            template = 'Epoch {}, Loss: {}'
            delta = prev_loss - self.train_loss.result().numpy()
            prev_loss = self.train_loss.result().numpy()

            print(template.format(done_epochs + 1,
                                  self.train_loss.result().numpy()))
            done_epochs += 1

        self.train_loss.reset_states()

        loss_object = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

        done_epochs = 0
        prev_loss = float('inf')
        delta = float('inf')

        while done_epochs < epochs and (prev_loss == -1 or delta > error_delta):
            for train_data in tqdm(train_ds, total=len(train_user_item_ratings) // batch_size):
                user_id, item_id, rating = train_data
                self.mlp_pretrain_step(mlp_model, optimizer, loss_object, user_id, item_id, rating)

            template = 'Epoch {}, Loss: {}'
            delta = prev_loss - self.train_loss.result().numpy()
            prev_loss = self.train_loss.result().numpy()

            print(template.format(done_epochs + 1,
                                  self.train_loss.result().numpy()))
            done_epochs += 1

        self.train_loss.reset_states()

        return gmf_model, mlp_model

    def fit(self, train_user_items_ratings, test_user_items_ratings=None, batch_size=20, epochs=20, early_stopping=-1):
        self.user_ratings = create_user_items_rating_matrix_w_indexer(train_user_items_ratings, self.indexer)

        train_user_items_ratings.extend(self._generate_negative_samples(train_user_items_ratings, 1))
        gmf_model, mlp_model = self._pretrain_models(train_user_items_ratings, epochs=100, error_delta=0.0001)

        self.model = self._build_neu_mf_model(pretrained_gmf_model=gmf_model, pretrained_mlp_model=mlp_model)

        train_ds = self._generate_dataset(train_user_items_ratings, batch_size)
        eval_test = test_user_items_ratings is not None

        if eval_test:
            test_ds = self._generate_dataset(test_user_items_ratings, batch_size)

        loss_object = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

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

        self.model.save(f'{CHECKPOINTS_DIRECTORY}/{self.name}.h5')

    def predict(self, user_id, item_id):
        user_input = np.array([self.indexer.get_user_internal_id(user_id)])
        item_input = np.array([self.indexer.get_user_internal_id(item_id)])
        return self.model.predict([user_input, item_input]).squeeze().tolist()

    def get_recommendations(self, user_id, k=None):
        user_input = np.full((self.num_items, 1), fill_value=self.indexer.get_user_internal_id(user_id))
        item_input = np.expand_dims(np.arange(0, self.num_items), axis=1)

        recommendations = self.model.predict([user_input, item_input]).squeeze()
        recommendations_idx = np.argsort(recommendations)[::-1]
        recommendations_idx = [self.indexer.get_movie_id(internal_id) for internal_id in recommendations_idx]
        return recommendations_idx[:k] if k is not None else recommendations_idx

