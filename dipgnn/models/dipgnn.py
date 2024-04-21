import tensorflow as tf
from dipgnn.models.model_utils import swish
from dipgnn.models.initializers import GlorotOrthogonal
from dipgnn.models.layers.embedding_layer import EmbeddingLayer
from dipgnn.models.layers.gaussian_expansion import GaussianExpansion
from dipgnn.models.layers.bessel_basis_layer import BesselBasisLayer
from dipgnn.models.layers.spherical_basis_layer import SphericalBasisLayer
from dipgnn.models.layers.bond2bond_block import Bond2BondBlock
from dipgnn.models.layers.atom2bond_block import Atom2BondBlock
from dipgnn.models.layers.bond2atom_block import Bond2AtomBlock
from dipgnn.models.layers.readout_layer import ReadoutLayer
from dipgnn.utils.register import registers


@registers.model.register("dipgnn")
class DiPGNN(tf.keras.Model):
    """
    DiPGNN model.
    """
    def __init__(
        self,
        atom_size,
        atom_embedding_size,
        bond_embedding_size,
        hidden_size,
        num_layers,
        name='dipgnn',
        logging=None,
        cutoff=5.0,
        activation=swish,
        target_type="atom",
        num_targets=12,
        atom_weight=0.9,
        bond_weight=0.1,
        kernel_initializer='zeros',
        rbf="Bessel",
        num_radial=6,
        envelope_exponent=5,
        sbf="Spherical",
        num_spherical=7,
        num_gaussian=50,
        gaussian_radial_var=0.2,
        gaussian_angular_var=0.2,
        use_extra_features=False,
        num_embedding_fc_layers=2,
        num_b2b_res_layers=2,
        num_readout_fc_layers=3,
        embedding_dropout=0,
        output_dropout=0,
        feature_add_or_concat="add",
    ):
        super().__init__(name=name)
        self.logging = logging
        self.num_layers = num_layers
        self.embedding_dropout = embedding_dropout
        self.output_dropout = output_dropout

        assert target_type in ['atom', 'bond', 'structure']
        self.target_type = target_type
        self.atom_weight = atom_weight
        self.bond_weight = bond_weight
        self.feature_add_or_concat = feature_add_or_concat
        if kernel_initializer == 'GlorotOrthogonal':
            kernel_initializer = GlorotOrthogonal()

        if rbf == "Bessel":
            self.rbf_layer = BesselBasisLayer(num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        elif rbf == "Gaussian":
            self.rbf_layer = GaussianExpansion(dmin=0.0, dmax=cutoff, num_gaussian=num_gaussian, var=gaussian_radial_var)
        else:
            raise ValueError("Rbf method {} isn't supported yet. We support ['Bessel', 'Gaussian'] method.".format(rbf))

        self.sbf = sbf
        if sbf == "Spherical":
            self.sbf_layer = SphericalBasisLayer(
                num_spherical, num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        elif sbf == "Gaussian":
            self.sbf_layer = GaussianExpansion(
                dmin=0.0, dmax=3.14, num_gaussian=num_gaussian, var=gaussian_angular_var)
        else:
            raise ValueError("Sbf method {} isn't supported yet. We support ['Spherical', 'Gaussian'] method.".format(sbf))

        self.embedding_layer = EmbeddingLayer(
            atom_size, atom_embedding_size, bond_embedding_size, use_extra_features=use_extra_features,
            num_embedding_fc_layers=num_embedding_fc_layers, activation=activation, kernel_initializer=kernel_initializer)

        self.bond2bond_blocks = list()
        self.bond2atom_blocks = list()
        self.atom2bond_blocks = list()
        for i_layer in range(num_layers):
            self.bond2bond_blocks.append(
                Bond2BondBlock(hidden_size, num_b2b_res_layers, activation=activation, kernel_initializer=kernel_initializer))
            self.bond2atom_blocks.append(
                Bond2AtomBlock(hidden_size, activation=activation, kernel_initializer=kernel_initializer))
            self.atom2bond_blocks.append(
                Atom2BondBlock(hidden_size, activation=activation, kernel_initializer=kernel_initializer))
        self.readout_layer = ReadoutLayer(
            hidden_size, num_readout_fc_layers, num_targets=num_targets, activation=activation, kernel_initializer=kernel_initializer)

    def call(self, inputs, validation_test=False):
        atom_features, indices_i, indices_j = \
            inputs['atom_features_list'], inputs['id_i_list'], inputs['id_j_list']
        dist_kji_kj_expand_to_angle, dist_kji_ji_expand_to_angle, angle_kji_reduce_to_dist = \
            inputs['dist_kj_expand_to_angle_list'], inputs['dist_kj_ji_expand_to_angle_list'], inputs['angle_kj_reduce_to_dist_list']
        dist_jim_im_expand_to_angle, dist_jim_ji_expand_to_angle, angle_jim_reduce_to_dist = \
            inputs['dist_im_expand_to_angle_list'], inputs['dist_im_ji_expand_to_angle_list'], inputs['angle_im_reduce_to_dist_list']
        distances, angles_kj, angles_im, reduce_to_target_indices = \
            inputs['dist_list'], inputs['angle_kj_list'], inputs['angle_im_list'], inputs['reduce_to_target_indices']

        rbf = self.rbf_layer(distances)

        if self.sbf == "Spherical":
            sbf_kj = self.sbf_layer(distances, angles_kj, dist_kji_kj_expand_to_angle)
            sbf_im = self.sbf_layer(distances, angles_im, dist_jim_im_expand_to_angle)
        elif self.sbf == "Gaussian":
            sbf_kj = self.sbf_layer(angles_kj)
            sbf_im = self.sbf_layer(angles_im)

        atom_embedding, bond_embedding = self.embedding_layer(atom_features, rbf, indices_i, indices_j)
        if not validation_test:
            atom_embedding = tf.nn.dropout(atom_embedding, rate=self.embedding_dropout)
            bond_embedding = tf.nn.dropout(bond_embedding, rate=self.embedding_dropout)

        if self.target_type == "atom":
            hidden_states = atom_embedding
        elif self.target_type == "bond":
            hidden_states = bond_embedding
        elif self.target_type == "structure":
            hidden_states = self.atom_weight * atom_embedding + self.bond_weight * tf.math.unsorted_segment_sum(
                bond_embedding, indices_i, tf.shape(atom_embedding)[0])
            hidden_states = tf.math.unsorted_segment_sum(hidden_states, inputs['reduce_to_target_indices'], inputs['n_structures'])
        else:
            raise ValueError("Only support target_type in ['atom', 'bond', 'structure']")

        for i_layer in range(self.num_layers):
            bond_embedding = self.bond2bond_blocks[i_layer](bond_embedding,
                sbf_kj, dist_kji_kj_expand_to_angle, dist_kji_ji_expand_to_angle, angle_kji_reduce_to_dist,
                sbf_im, dist_jim_im_expand_to_angle, dist_jim_ji_expand_to_angle, angle_jim_reduce_to_dist)
            if not validation_test:
                bond_embedding = tf.nn.dropout(bond_embedding, rate=self.output_dropout)

            atom_embedding = self.bond2atom_blocks[i_layer](atom_embedding, bond_embedding, indices_i, indices_j)
            bond_embedding = self.atom2bond_blocks[i_layer](atom_embedding, bond_embedding, indices_i, indices_j)
            if not validation_test:
                bond_embedding = tf.nn.dropout(bond_embedding, rate=self.output_dropout)

            if self.target_type == "structure":
                hidden_states = self.atom_weight * atom_embedding + self.bond_weight * tf.math.unsorted_segment_sum(
                    bond_embedding, indices_i, tf.shape(atom_embedding)[0])
                if self.feature_add_or_concat == "add":
                    hidden_states += tf.math.unsorted_segment_sum(hidden_states, inputs['reduce_to_target_indices'], inputs['n_structures'])
                elif self.feature_add_or_concat == "concat":
                    hidden_states = tf.concat([hidden_states, tf.math.unsorted_segment_sum(hidden_states, inputs['reduce_to_target_indices'], inputs['n_structures'])], 1)
                else:
                    raise ValueError("feature_add_or_concat only supports add or concat")
            else:
                if self.target_type == "atom":
                    if self.feature_add_or_concat == "add":
                        hidden_states += atom_embedding
                    else:
                        hidden_states = tf.concat([hidden_states, atom_embedding], 1)
                elif self.target_type == "bond":
                    if self.feature_add_or_concat == "add":
                        hidden_states += bond_embedding
                    else:
                        hidden_states = tf.concat([hidden_states, bond_embedding], 1)

        outputs = self.readout_layer(hidden_states)

        if self.target_type == "structure":
            return outputs
        else:
            return tf.gather(outputs, reduce_to_target_indices)
