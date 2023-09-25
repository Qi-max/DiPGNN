import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class EmbeddingLayer(layers.Layer):
    def __init__(
        self,
        atom_size,
        atom_embedding_size,
        bond_embedding_size,
        name='embedding_layer',
        activation=None,
        use_extra_features=False,
        num_embedding_fc_layers=2,
        kernel_initializer="glorot_uniform"
    ):
        super().__init__(name=name)
        self.use_extra_features = use_extra_features
        if use_extra_features:
            self.atom_embedding_layers = list()
            for i in range(num_embedding_fc_layers):
                self.atom_embedding_layers.extend([
                    layers.Dense(atom_embedding_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
                    layers.BatchNormalization()])
            self.atom_embedding_layers = Sequential(self.atom_embedding_layers)
        else:
            self.atom_embedding_table = tf.get_variable(
                "atom_embedding_table", [atom_size, atom_embedding_size],
                initializer=tf.keras.initializers.glorot_normal())

        self.bond_preprocess_fc_layer = Sequential([
            layers.Dense(bond_embedding_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

        self.bond_embedding_layer = Sequential([
            layers.Dense(bond_embedding_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

    def call(self, atom_features, bond_features, indices_i, indices_j):
        atom_embedding = atom_features
        if self.use_extra_features:
            atom_embedding = self.atom_embedding_layers(atom_embedding)
        else:
            atom_embedding = tf.nn.embedding_lookup(self.atom_embedding_table, atom_embedding)

        atom_embedding_i = tf.gather(atom_embedding, indices_i)
        atom_embedding_j = tf.gather(atom_embedding, indices_j)

        bond_embedding = self.bond_preprocess_fc_layer(bond_features)
        bond_embedding = self.bond_embedding_layer(tf.concat([atom_embedding_i, bond_embedding, atom_embedding_j], axis=-1))
        return atom_embedding, bond_embedding
