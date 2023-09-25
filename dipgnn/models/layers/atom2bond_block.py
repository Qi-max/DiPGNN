import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


class Atom2BondBlock(layers.Layer):
    def __init__(
        self,
        hidden_size,
        name='atom2bond_block',
        activation=None,
        kernel_initializer="glorot_uniform"
    ):
        super().__init__(name=name)
        self.bond_fc_layers = Sequential([
            layers.Dense(hidden_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
            layers.BatchNormalization(),
            layers.Dense(hidden_size, activation=activation, use_bias=True, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

    def call(self, atom_embedding, bond_embedding, indices_i, indices_j):
        atom_embedding_i = tf.gather(atom_embedding, indices_i)
        atom_embedding_j = tf.gather(atom_embedding, indices_j)
        bond_embedding += self.bond_fc_layers(tf.concat([atom_embedding_i, bond_embedding, atom_embedding_j], axis=-1))
        return bond_embedding
