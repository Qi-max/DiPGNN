import os
import pickle
import logging
import string
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from dipgnn.models.model_utils import swish
from dipgnn.utils.classification_metrics import ClassificationMetrics
from dipgnn.utils.register import registers


@registers.task.register("classification_training_task")
class ClassificationTrainingTask:
    def __int__(
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

    def initial_paths(self):
        # Used for creating a random "unique" id for this run
        def id_generator(size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
            return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

        # Create output path. A unique directory name is created for this run based on the input
        if self.args.ckpt_path is None:
            self.output_path = "{}/{}_{}_{}{}".format(
                self.args.base_output_path, datetime.now().strftime("%Y%m%d_%H%M%S"),
                id_generator(), self.args.target_name, self.args.comment)
        else:
            self.output_path = self.args.ckpt_path
        logging.info(f"self.output_path: {self.output_path}")

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

    def initial_metrics(self):
        self.training_metrics = ClassificationMetrics('train', ["targets"])
        self.validation_metrics = ClassificationMetrics('validation', ["targets"])
        self.test_metrics = ClassificationMetrics('test', ["targets"])

    def initial_data_container(self):
        graph_data_file_list = pd.read_csv(self.args.graph_data_file_df, index_col=0).values
        logging.info("file length: {}".format(len(graph_data_file_list)))

        self.data_container = registers.data_container.get_class(args.data_container_name).from_files(
            graph_data_file_list=graph_data_file_list,
            targets_data_file=self.args.targets_data_file,
            target_type=self.args.target_type,
            task=self.args.task,
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
        logging.info("model summary is: {}".format(self.model.summary()))

    def initial_trainer(self):
        self.trainer = registers.model.get_class(self.args.trainer_name)(
            model=self.model,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            decay_steps=self.args.decay_steps,
            decay_rate=self.args.decay_rate,
            ema_decay=self.args.ema_decay,
            max_grad_norm=self.max_grad_norm)
        logging.info("Initial trainer end.")

    def run(self):
        num_train, num_validation, num_test = self.data_provider.get_train_validation_test_num()

        if os.path.isfile(self.best_loss_file):
            loss_file = np.load(self.best_loss_file)
            metrics_best = {k: v.item() for k, v in loss_file.items()}
        else:
            metrics_best = self.validation_metrics.result()
            for key in metrics_best.keys():
                metrics_best[key] = 0
            metrics_best['step'] = 0
            metrics_best['epoch'] = 0
            np.savez(self.best_loss_file, **metrics_best)

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.trainer.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.logger_path, max_to_keep=self.args.ckpt_max_to_keep)

        # Restore latest checkpoint
        ckpt_restored = tf.train.latest_checkpoint(self.logger_path)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored)

        # save normalizer
        if self.data_container.target_normalizer is not None:
            with open(os.path.join(self.best_path, 'normalizer.pkl'), "wb") as normalizer_file:
                pickle.dump(self.data_container.target_normalizer, normalizer_file, protocol=4)

        # save args
        with open(os.path.join(self.best_path, 'args.pkl'), "wb") as args_file:
            pickle.dump({"args": vars(self.args)}, args_file, protocol=4)

        with self.summary_writer.as_default():
            steps_per_epoch = int(np.ceil(num_train / self.args.batch_size))
            step_initial = 1 if ckpt_restored is not None else ckpt.step.numpy()

            train_targets_list = list()
            train_preds_list = list()
            first_save_model = True

            for step in range(step_initial, self.args.num_steps + 1):
                # Update step number
                ckpt.step.assign(step)
                tf.summary.experimental.set_step(step)

                # Perform training step
                _, train_targets, train_preds = self.trainer.train_on_batch(
                    self.train_dataset,
                    self.training_metrics,
                    l2_loss_decay=self.args.l2_loss_decay,
                    use_huber_loss=self.args.use_huber_loss,
                    huber_delta=self.args.huber_delta)

                train_targets_list += train_targets[:, 0].numpy().tolist()
                train_preds_list += train_preds[:, 0].numpy().tolist()

                if step % steps_per_epoch == 0:
                    manager.save()

                    train_targets_list = list()
                    train_preds_list = list()
                    validation_targets_list = list()
                    validation_preds_list = list()
                    test_targets_list = list()
                    test_preds_list = list()

                    # Save backup variables and load averaged variables
                    self.trainer.save_variable_backups()
                    self.trainer.load_averaged_variables()

                    # Compute results on the validation set
                    for i in range(int(np.ceil(num_validation / self.args.batch_size))):
                        _, validation_targets, validation_preds = self.trainer.test_on_batch(
                            self.validation_dataset, self.validation_metrics)
                        validation_targets_list += validation_targets[:, 0].numpy().tolist()
                        validation_preds_list += validation_preds[:, 0].numpy().tolist()
                    self.trainer.load_averaged_variables()

                    # Compute results on the test set
                    for i in range(int(np.ceil(num_test / self.args.batch_size))):
                        _, test_targets, test_preds = self.trainer.test_on_batch(self.test_dataset, self.test_metrics)
                        test_targets_list += test_targets[:, 0].numpy().tolist()
                        test_preds_list += test_preds[:, 0].numpy().tolist()

                    epoch = step // steps_per_epoch
                    # Update and save best result
                    if self.validation_metrics.mean_auc > metrics_best['mean_auc_validation']:
                        metrics_best['epoch'] = epoch
                        metrics_best.update(self.validation_metrics.result())
                        np.savez(self.best_loss_file, **metrics_best)

                        if first_save_model:
                            self.model.save(os.path.join(self.best_path, 'best-full-model-epoch{}'.format(epoch)))
                            first_save_model = False

                        if self.args.save_all_models:
                            self.model.save_weights(os.path.join(self.best_path, 'best-model-epoch{}'.format(epoch)))
                        else:
                            self.model.save_weights(os.path.join(self.best_path, 'best-model'))

                        if self.args.save_all_predictions:
                            pd.DataFrame({
                                "validation_targets": validation_targets_list,
                                "validation_preds": validation_preds_list,
                            }).to_csv(os.path.join(self.best_path, 'best_predict_validation_epoch{}.csv'.format(epoch)))

                            pd.DataFrame({
                                "test_targets": test_targets_list,
                                "test_preds": test_preds_list,
                            }).to_csv(os.path.join(self.best_path, 'best_predict_test_epoch{}.csv'.format(epoch)))
                        else:
                            pd.DataFrame({
                                "validation_targets": validation_targets_list,
                                "validation_preds": validation_preds_list,
                            }).to_csv(os.path.join(self.best_path, 'best_predict_validation.csv'))

                            pd.DataFrame({
                                "test_targets": test_targets_list,
                                "test_preds": test_preds_list,
                            }).to_csv(os.path.join(self.best_path, 'best_predict_test.csv'))

                        with open(os.path.join(self.best_path, 'best_logger.txt'), "a") as file:
                            file.write(
                                f"{step}/{self.args.num_steps} (epoch {epoch}): "
                                f"Loss: train={self.training_metrics.loss:.6f}, "
                                    f"validation={self.validation_metrics.loss:.6f}, "
                                    f"test={self.test_metrics.loss:.6f}; "
                                f"ACC: train={self.training_metrics.mean_acc:.6f}, "
                                    f"validation={self.validation_metrics.mean_acc:.6f}, "
                                    f"test={self.test_metrics.mean_acc:.6f};"
                                f"RECALL: train={self.training_metrics.mean_recall:.6f}, "
                                    f"validation={self.validation_metrics.mean_recall:.6f}, "
                                    f"test={self.test_metrics.mean_recall:.6f};"
                                f"PRECISION: train={self.training_metrics.mean_precision:.6f}, "
                                    f"validation={self.validation_metrics.mean_precision:.6f}, "
                                    f"test={self.test_metrics.mean_precision:.6f};"
                                f"AUC: train={self.training_metrics.mean_auc:.6f}, "
                                    f"validation={self.validation_metrics.mean_auc:.6f}, "
                                    f"test={self.test_metrics.mean_auc:.6f};"
                                f"F1: train={self.training_metrics.mean_f1:.6f}, "
                                    f"validation={self.validation_metrics.mean_f1:.6f}, "
                                    f"test={self.test_metrics.mean_f1:.6f}.\n")

                        with open(os.path.join(self.best_path, 'best_scores.csv'), "a") as file:
                            file.write(
                                f"{step},{epoch},{self.training_metrics.loss:.6f},{self.validation_metrics.loss:.6f},{self.test_metrics.loss:.6f},"
                                f"{self.training_metrics.mean_acc:.6f},{self.validation_metrics.mean_acc:.6f},{self.test_metrics.mean_acc:.6f},"
                                f"{self.training_metrics.mean_recall:.6f},{self.validation_metrics.mean_recall:.6f},{self.test_metrics.mean_recall:.6f},"
                                f"{self.training_metrics.mean_precision:.6f},{self.validation_metrics.mean_precision:.6f},{self.test_metrics.mean_precision:.6f},"
                                f"{self.training_metrics.mean_auc:.6f},{self.validation_metrics.mean_auc:.6f},{self.test_metrics.mean_auc:.6f},"
                                f"{self.training_metrics.mean_f1:.6f},{self.validation_metrics.mean_f1:.6f},{self.test_metrics.mean_f1:.6f}\n")

                    else:
                        if epoch > metrics_best['epoch'] + self.args.early_stopping_epochs:
                            logging.info(f"Breaking the training as the score does not increase"
                                         f"for {self.args.early_stopping_epochs} epochs. Best epoch: {metrics_best['epoch']}")
                            break

                    for key, validation in metrics_best.items():
                        if key != 'step':
                            tf.summary.scalar(key + '_best', validation)

                    logging.info(
                        f"{step}/{self.args.num_steps} (epoch {epoch}): "
                        f"Loss: train={self.training_metrics.loss:.6f}, "
                            f"validation={self.validation_metrics.loss:.6f}, "
                            f"test={self.test_metrics.loss:.6f}; "
                        f"ACC: train={self.training_metrics.mean_acc:.6f}, "
                            f"validation={self.validation_metrics.mean_acc:.6f}, "
                            f"test={self.test_metrics.mean_acc:.6f};"
                        f"RECALL: train={self.training_metrics.mean_recall:.6f}, "
                            f"validation={self.validation_metrics.mean_recall:.6f}, "
                            f"test={self.test_metrics.mean_recall:.6f};"
                        f"PRECISION: train={self.training_metrics.mean_precision:.6f}, "
                            f"validation={self.validation_metrics.mean_precision:.6f}, "
                            f"test={self.test_metrics.mean_precision:.6f};"
                        f"AUC: train={self.training_metrics.mean_auc:.6f}, "
                            f"validation={self.validation_metrics.mean_auc:.6f}, "
                            f"test={self.test_metrics.mean_auc:.6f};"
                        f"F1: train={self.training_metrics.mean_f1:.6f}, "
                            f"validation={self.validation_metrics.mean_f1:.6f}, "
                            f"test={self.test_metrics.mean_f1:.6f}\n")

                    self.training_metrics.write()
                    self.validation_metrics.write()
                    self.test_metrics.write()

                    self.training_metrics.reset_states()
                    self.validation_metrics.reset_states()
                    self.test_metrics.reset_states()

                    # Restore backup variables
                    self.trainer.restore_variable_backups()
