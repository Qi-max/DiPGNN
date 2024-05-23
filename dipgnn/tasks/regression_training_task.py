import os
import pickle
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
from dipgnn.utils.register import registers
from dipgnn.tasks.base_training_task import BaseTrainingTask
from dipgnn.utils.regression_metrics import RegressionMetrics


@registers.task.register("regression_training_task")
class RegressionTrainingTask(BaseTrainingTask):
    def __init__(
        self,
        args
    ):
        super().__init__(args=args)

    def initial_metrics(self):
        self.training_metrics = RegressionMetrics('train', ["targets"])
        self.validation_metrics = RegressionMetrics('validation', ["targets"])
        self.test_metrics = RegressionMetrics('test', ["targets"])

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
            first_save_model = True

            for step in range(step_initial, self.args.num_steps + 1):
                # Update step number
                ckpt.step.assign(step)
                tf.summary.experimental.set_step(step)

                # Perform training step
                self.trainer.train_on_batch(self.train_dataset, self.training_metrics)

                if step % steps_per_epoch == 0:
                    manager.save()

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
                        validation_targets_list += self.data_container.target_normalizer.inverse_transform(validation_targets[:, 0].numpy().reshape(-1, 1)).reshape(-1).tolist()
                        validation_preds_list += self.data_container.target_normalizer.inverse_transform(validation_preds[:, 0].numpy().reshape(-1, 1)).reshape(-1).tolist()

                    # Compute results on the test set
                    self.trainer.load_averaged_variables()
                    for i in range(int(np.ceil(num_test / self.args.batch_size))):
                        _, test_targets, test_preds = self.trainer.test_on_batch(self.test_dataset, self.test_metrics)
                        test_targets_list += self.data_container.target_normalizer.inverse_transform(test_targets[:, 0].numpy().reshape(-1, 1)).reshape(-1).tolist()
                        test_preds_list += self.data_container.target_normalizer.inverse_transform(test_preds[:, 0].numpy().reshape(-1, 1)).reshape(-1).tolist()

                    epoch = step // steps_per_epoch
                    # Update and save best result
                    if self.validation_metrics.mean_r2 > metrics_best['mean_r2_validation']:
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
                                f"MAE: train={self.training_metrics.mean_mae:.6f}, "
                                    f"validation={self.validation_metrics.mean_mae:.6f}, "
                                    f"test={self.test_metrics.mean_mae:.6f}; "
                                f"MSE: train={self.training_metrics.mean_mse:.6f}, "
                                    f"validation={self.validation_metrics.mean_mse:.6f}, "
                                    f"test={self.test_metrics.mean_mse:.6f}; "
                                f"RMSE: train={self.training_metrics.mean_rmse:.6f}, "
                                    f"validation={self.validation_metrics.mean_rmse:.6f}, "
                                    f"test={self.test_metrics.mean_rmse:.6f}; "
                                f"R2: train={self.training_metrics.mean_r2:.6f}, "
                                    f"validation={self.validation_metrics.mean_r2:.6f}, "
                                    f"test={self.test_metrics.mean_r2:.6f}; "
                                f"Pearson: train={self.training_metrics.mean_pearson:.6f}, "
                                    f"validation={self.validation_metrics.mean_pearson:.6f}, "
                                    f"test={self.test_metrics.mean_pearson:.6f}.\n")

                        with open(os.path.join(self.best_path, 'best_scores.csv'), "a") as file:
                            file.write(
                                f"{step},{epoch},{self.training_metrics.loss:.6f},{self.validation_metrics.loss:.6f},{self.test_metrics.loss:.6f},"
                                f"{self.training_metrics.mean_mae:.6f},{self.validation_metrics.mean_mae:.6f},{self.test_metrics.mean_mae:.6f},"
                                f"{self.training_metrics.mean_mse:.6f},{self.validation_metrics.mean_mse:.6f},{self.test_metrics.mean_mse:.6f},"
                                f"{self.training_metrics.mean_rmse:.6f},{self.validation_metrics.mean_rmse:.6f},{self.test_metrics.mean_rmse:.6f},"
                                f"{self.training_metrics.mean_r2:.6f},{self.validation_metrics.mean_r2:.6f},{self.test_metrics.mean_r2:.6f},"
                                f"{self.training_metrics.mean_pearson:.6f},{self.validation_metrics.mean_pearson:.6f},{self.test_metrics.mean_pearson:.6f}\n")

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
                        f"MAE: train={self.training_metrics.mean_mae:.6f}, "
                            f"validation={self.validation_metrics.mean_mae:.6f}, "
                            f"test={self.test_metrics.mean_mae:.6f}; "
                        f"MSE: train={self.training_metrics.mean_mse:.6f}, "
                            f"validation={self.validation_metrics.mean_mse:.6f}, "
                            f"test={self.test_metrics.mean_mse:.6f}; "
                        f"RMSE: train={self.training_metrics.mean_rmse:.6f}, "
                            f"validation={self.validation_metrics.mean_rmse:.6f}, "
                            f"test={self.test_metrics.mean_rmse:.6f}; "
                        f"R2: train={self.training_metrics.mean_r2:.6f}, "
                            f"validation={self.validation_metrics.mean_r2:.6f}, "
                            f"test={self.test_metrics.mean_r2:.6f}; "
                        f"Pearson: train={self.training_metrics.mean_pearson:.6f}, "
                            f"validation={self.validation_metrics.mean_pearson:.6f}, "
                            f"test={self.test_metrics.mean_pearson:.6f}.\n")

                    self.training_metrics.write()
                    self.validation_metrics.write()
                    self.test_metrics.write()

                    self.training_metrics.reset_states()
                    self.validation_metrics.reset_states()
                    self.test_metrics.reset_states()

                    # Restore backup variables
                    self.trainer.restore_variable_backups()
                if step == step_initial:
                    self.model.summary()
