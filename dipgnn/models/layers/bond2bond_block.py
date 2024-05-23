import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from dipgnn.models.layers.residual_layer import ResidualLayer


class Bond2BondBlock(layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_b2b_res_layers,
        name='bond2bond_block',
        activation=None,
        kernel_initializer="glorot_uniform"
    ):
        super().__init__(name=name)
        self.bond_kj_fc_layers = Sequential([
            layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization(),
            layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

        self.angle_ijk_attention_layers = Sequential([
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer)])

        self.bond_im_fc_layers = Sequential([
            layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization(),
            layers.Dense(hidden_size + 1, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

        self.angle_ijm_attention_layers = Sequential([
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer)])

        self.bond_preprocess_fc_layer = Sequential([
            layers.Dense(hidden_size, activation=activation, use_bias=False, kernel_initializer=kernel_initializer),
            layers.BatchNormalization()])

        self.residual_layers = list()
        for i in range(num_b2b_res_layers):
            self.residual_layers.append(ResidualLayer(
                hidden_size, activation=activation, use_bias=True,
                kernel_initializer=kernel_initializer, name="b2b_residual_layer_{}".format(i)))
        self.residual_layers = Sequential(self.residual_layers)

    def call(self, bond_embedding,
             sbf_mij, bond_mi_id_for_angle_mij_list, bond_ij_id_for_angle_mij_list,
             sbf_kji, bond_kj_id_for_angle_kji_list, bond_ij_id_for_angle_kji_list,):

        bond_embedding_mij_mi = tf.gather(bond_embedding, bond_mi_id_for_angle_mij_list)
        bond_embedding_mij_ij = tf.gather(bond_embedding, bond_ij_id_for_angle_mij_list)
        bond_mij_updated = self.bond_im_fc_layers(tf.concat([bond_embedding_mij_mi, bond_embedding_mij_ij], axis=-1))
        angle_mij_attentions = self.angle_ijm_attention_layers(sbf_mij)

        bond_embedding_kji_kj = tf.gather(bond_embedding, bond_kj_id_for_angle_kji_list)
        bond_embedding_kji_ij = tf.gather(bond_embedding, bond_ij_id_for_angle_kji_list)
        bond_kji_updated = self.bond_kj_fc_layers(tf.concat([bond_embedding_kji_kj, bond_embedding_kji_ij], axis=-1))
        angle_kji_attentions = self.angle_ijk_attention_layers(sbf_kji)

        num_bonds = tf.shape(bond_embedding)[0]
        bond_embedding += \
            self.bond_preprocess_fc_layer(tf.concat([
                tf.math.unsorted_segment_sum(angle_mij_attentions * bond_mij_updated[:, :1] * bond_mij_updated[:, 1:], bond_ij_id_for_angle_mij_list, num_bonds),
                tf.math.unsorted_segment_sum(angle_kji_attentions * bond_kji_updated[:, :1] * bond_kji_updated[:, 1:], bond_ij_id_for_angle_kji_list, num_bonds),
            ], axis=-1))
        bond_embedding = self.residual_layers(bond_embedding)
        return bond_embedding
