import os
import logging
import string
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from dipgnn.models.model_utils import swish
from dipgnn.utils.register import registers
from abc import ABC, abstractmethod


@registers.task.register("base_training_task")
class BaseTrainingTask(ABC):
    def __init__(
        self,
        args
    ):
        self.args = args
        self.initial_task_environment()

    def initial_task_environment(self):
        self.initial_paths()
        self.initial_logger()
        self.initial_metrics()
        self.initial_data_container()
        self.initial_data_provider()
        self.initial_dataset()
        self.initial_model()
        self.initial_trainer()

    def initial_paths(self):
        # Used for creating a random "unique" id for this run
        def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
            return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

        # Create output path. A unique directory name is created for this run based on the input
        if self.args.ckpt_path is None:
            self.output_path = "{}/{}_{}_{}{}".format(
                self.args.output_path, datetime.now().strftime("%Y%m%d_%H%M%S"),
                id_generator(), self.args.target_name, self.args.comment)
        else:
            self.output_path = self.args.ckpt_path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.best_path = os.path.join(self.output_path, 'best')
        if not os.path.exists(self.best_path):
            os.makedirs(self.best_path)
        self.logger_path = os.path.join(self.output_path, 'logs')
        if not os.path.exists(self.logger_path):
            os.makedirs(self.logger_path)
        self.best_loss_file = os.path.join(self.best_path, 'best_loss.npz')

        with open(os.path.join(self.best_path, 'best_logger.txt'), "a") as file:
            file.write(f"Model args is: {self.args}\n")

    def initial_logger(self):
        # local logger
        logger = logging.getLogger()
        formatter = logging.Formatter(fmt='%(asctime)s (%(levelname)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        fh = logging.FileHandler(os.path.join(self.logger_path, 'logging.log'), encoding='UTF-8')
        fh.setFormatter(formatter)

        logger.handlers = list()
        logger.addHandler(ch)
        logger.addHandler(fh)
        logger.setLevel('INFO')

        # tensorflow logger
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.get_logger().setLevel('WARN')
        tf.autograph.set_verbosity(2)
        self.summary_writer = tf.summary.create_file_writer(self.logger_path)
        logging.info(f"Output path is: {self.output_path}")

    @abstractmethod
    def initial_metrics(self):
        """Derived classes should implement this function."""

    def initial_data_container(self):
        graph_data_file_list = pd.read_csv(self.args.graph_data_file_df, index_col=0).values
        logging.info("file length: {}".format(len(graph_data_file_list)))

        self.data_container = registers.data_container.get_class(self.args.data_container_name).from_files(
            graph_data_file_list=graph_data_file_list,
            targets_data_file=self.args.targets_data_file,
            target_type=self.args.target_type,
            task_type=self.args.task_type,
            target_col=self.args.target_col)
        logging.info("Initial data_container end.")

    def initial_dataset(self):
        self.train_dataset = iter(self.data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE))
        self.validation_dataset = iter(self.data_provider.get_dataset('validation').prefetch(tf.data.experimental.AUTOTUNE))
        self.test_dataset = iter(self.data_provider.get_dataset('test').prefetch(tf.data.experimental.AUTOTUNE))

        # save train/validation/test indices
        np.save(os.path.join(self.output_path, 'train_indices.npy'), np.array(self.data_provider.idx["train"]))
        np.save(os.path.join(self.output_path, 'validation_indices.npy'), np.array(self.data_provider.idx["validation"]))
        np.save(os.path.join(self.output_path, 'test_indices.npy'), np.array(self.data_provider.idx["test"]))

    def initial_data_provider(self):
        self.data_provider = registers.data_provider.get_class(self.args.data_provider_name)(
            self.data_container,
            train=self.args.train,
            validation=self.args.validation,
            test=self.args.test,
            batch_size=self.args.batch_size,
            random_seed=self.args.random_seed,
            shuffle=self.args.shuffle,
            logging=logging)
        logging.info("Initial data_provider end.")

    def initial_model(self):
        self.model = registers.model.get_class(self.args.model_name)(
            target_type=self.args.target_type,
            cutoff=self.args.cutoff,
            atom_size=self.args.input_size,
            atom_embedding_size=self.args.atom_embedding_size,
            bond_embedding_size=self.args.bond_embedding_size,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            num_spherical=self.args.num_spherical,
            num_radial=self.args.num_radial,
            rbf=self.args.rbf,
            sbf=self.args.sbf,
            num_gaussian=self.args.num_gaussian,
            gaussian_radial_var=self.args.gaussian_radial_var,
            gaussian_angular_var=self.args.gaussian_angular_var,
            envelope_exponent=self.args.envelope_exponent,
            num_embedding_fc_layers=self.args.num_embedding_fc_layers,
            num_b2b_res_layers=self.args.num_b2b_res_layers,
            num_readout_fc_layers=self.args.num_readout_fc_layers,
            num_targets=self.args.num_targets,
            activation=swish,
            kernel_initializer=self.args.kernel_initializer,
            embedding_dropout=self.args.embedding_dropout,
            output_dropout=self.args.output_dropout,
            use_extra_features=True if self.data_container.atom_feature_scheme == "external" else False,
            feature_add_or_concat=self.args.feature_add_or_concat,
            logging=logging)
        logging.info("Initial model end.")

    def initial_trainer(self):
        self.trainer = registers.task.get_class(self.args.trainer_name)(
            model=self.model,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            decay_steps=self.args.decay_steps,
            decay_rate=self.args.decay_rate,
            ema_decay=self.args.ema_decay,
            max_grad_norm=self.args.max_grad_norm)
        logging.info("Initial trainer end.")

    @abstractmethod
    def run(self):
        """Derived classes should implement this function."""
