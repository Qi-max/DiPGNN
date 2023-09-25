import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from dipgnn.models.layers.residual_layer import ResidualLayer


class Bond2AtomBlock(layers.Layer):
    def __init__(
        self,
        hidden_size,
        name='bond2atom_block',
        activation=None,
        kernel_initializer='zeros'
    ):
        super().__init__(name=name)

        self.atom_preprocess_fc_layers = Sequential([
            layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization(),
            layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

        self.atom_residual_layers = Sequential([
            ResidualLayer(hidden_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
            ResidualLayer(hidden_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer)])

    def call(self, atom_embedding, bond_embedding, indices_i, indices_j):
        atom_embedding_i = tf.gather(atom_embedding, indices_i)
        atom_embedding_j = tf.gather(atom_embedding, indices_j)

        atom_embedding_updated = self.atom_preprocess_fc_layers(
            tf.concat([atom_embedding_i, bond_embedding, atom_embedding_j], axis=-1))
        atom_embedding_i = atom_embedding_updated[:, :1] * atom_embedding_updated[:, 1:]
        atom_embedding += tf.math.unsorted_segment_sum(atom_embedding_i, indices_i, tf.shape(atom_embedding)[0])

        atom_embedding = self.atom_residual_layers(atom_embedding)
        return atom_embedding
