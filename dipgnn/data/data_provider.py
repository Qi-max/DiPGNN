import random
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from dipgnn.utils.register import registers


@registers.data_provider.register("data_provider")
class DataProvider:
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """
    def __init__(
        self,
        data_container,
        train,
        validation,
        test,
        batch_size=1,
        random_seed=None,
        shuffle=False,
        logging=None
    ):
        self.data_container = data_container
        self.n_data = len(data_container)
        self.batch_size = batch_size

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=random_seed)
        self.logging = logging

        if isinstance(random_seed, int):
            random.seed(random_seed)
            tf.random.set_seed(random_seed)

        (train_indices, validation_indices, test_indices), (self.n_train, self.n_validation, self.n_test) = \
            get_train_validation_test_indices(
                total_size=self.n_data, train=train, validation=validation, test=test, shuffle=shuffle)

        logging.info("number of training files: {}\n".format(self.n_train))
        logging.info("training indices: {}\n".format(train_indices))

        logging.info("number of validation files: {}\n".format(self.n_validation))
        logging.info("validation indices: {}".format(validation_indices))

        logging.info("number of test files: {}\n".format(self.n_test))
        logging.info("test indices: {}".format(test_indices))

        # Store indices of training, validation and test data
        self.idx = {'train': train_indices,
                    'validation': validation_indices,
                    'test': test_indices}

        self.nsamples = {'train': self.n_train, 'validation': self.n_validation, 'test': self.n_test}

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'validation': 0, 'test': 0}

        # dtypes of dataset values
        self.dtypes_input = OrderedDict()

        for int_key in self.data_container.int_keys():
            self.dtypes_input[int_key] = tf.int32

        for float_key in self.data_container.float_keys():
            self.dtypes_input[float_key] = tf.float32

        for number_key in self.data_container.int_number_keys():
            self.dtypes_input[number_key] = tf.int32

        if data_container.atom_feature_scheme == "external":
            self.dtypes_input['atom_features_list'] = tf.float32
        elif data_container.atom_feature_scheme == "specie_onehot":
            self.dtypes_input['atom_features_list'] = tf.int32

        self.dtype_target = tf.float32

        # Shapes of dataset values
        self.shapes_input = dict()

        for int_key in self.data_container.int_keys():
            self.shapes_input[int_key] = [None]
        for float_key in self.data_container.float_keys():
            self.shapes_input[float_key] = [None]
        for number_key in self.data_container.int_number_keys():
            self.shapes_input[number_key] = None

        if data_container.atom_feature_scheme == "external":
            self.shapes_input['atom_features_list'] = [None, data_container.atom_feature_len]
        elif data_container.atom_feature_scheme == "specie_onehot":
            self.shapes_input['atom_features_list'] = [None]   # 注意：specie_onehot feature的内部没有小括号

        self.shape_target = [None, 1 if data_container.task_type == "regression" else 2]

    def get_train_validation_test_num(self):
        """get_train_validation_test_num"""
        return self.n_train, self.n_validation, self.n_test

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split):
        """Return the indices for a batch of samples from the specified set"""
        start = self.idx_in_epoch[split]

        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]

    def idx_to_data(self, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            inputs_targets.append(tf.constant(batch['targets'], dtype=self.dtype_target))
            return inputs_targets
        else:
            inputs = {}
            for key, dtype in self.dtypes_input.items():
                inputs[key] = tf.constant(batch[key], dtype=dtype)
            targets = tf.constant(batch['targets'], dtype=tf.float32)
            return (inputs, targets)

    def get_dataset(self, split):
        """Get a generator-based tf.dataset"""
        def generator():
            while True:
                idx = self.get_batch_idx(split)
                yield self.idx_to_data(idx)

        return tf.data.Dataset.from_generator(
            generator,
            output_types=(dict(self.dtypes_input), self.dtype_target),
            output_shapes=(self.shapes_input, self.shape_target))

    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)

        return tf.data.Dataset.from_generator(
            generator,
            output_types=tf.int32,
            output_shapes=[None])

    def idx_to_data_tf(self, idx):
        """Convert a batch of indices to a batch of data from TensorFlow"""
        dtypes_flattened = list(self.dtypes_input.values())
        dtypes_flattened.append(self.dtype_target)

        inputs_targets = tf.py_function(lambda idx: self.idx_to_data(idx.numpy(), return_flattened=True),
                                        inp=[idx], Tout=dtypes_flattened)

        inputs = {}
        for i, key in enumerate(self.dtypes_input.keys()):
            inputs[key] = inputs_targets[i]
            inputs[key].set_shape(self.shapes_input[key])
        targets = inputs_targets[-1]
        targets.set_shape(self.shape_target)
        return (inputs, targets)


def get_train_validation_test_indices(
        total_size,
        train=0.8,
        validation=0.1,
        test=0.1,
        shuffle=True
):

    if all((train.endswith("npy"), validation.endswith("npy"), validation.endswith("npy"))):
        train_indices, validation_indices, test_indices = np.load(train), np.load(validation), np.load(test)
        train_size, validation_size, test_size = len(train_indices), len(validation_indices), len(test_indices)

    else:
        train, validation, test = eval(train), eval(validation), eval(test)

        if all((isinstance(train, (list, tuple)),
                isinstance(validation, (list, tuple)),
                isinstance(test, (list, tuple)))):
            train_indices, validation_indices, test_indices = train, validation, test
            train_size, validation_size, test_size = len(train), len(validation), len(test)

        else:
            if all((isinstance(train, float),
                    isinstance(validation, float),
                    isinstance(test, float))):
                assert train + validation + test <= 1
                validation_size = int(np.ceil(validation * total_size))
                test_size = int(np.ceil(test * total_size))
                train_size = min(int(np.ceil(train * total_size)), total_size - validation_size - test_size)

            elif all((isinstance(train, int),
                      isinstance(validation, int),
                      isinstance(test, int))):
                assert train + validation + test <= total_size
                train_size, validation_size, test_size = train, validation, test

            else:
                raise ValueError("Please provide train/validation/test_size as float, int or list-like array")

            indices = list(range(total_size))
            if shuffle:
                indices = random.sample(indices, total_size)

            train_indices = indices[:train_size]
            validation_indices = indices[-(validation_size + test_size):-test_size]
            test_indices = indices[-test_size:]

    return (train_indices, validation_indices, test_indices), (train_size, validation_size, test_size)