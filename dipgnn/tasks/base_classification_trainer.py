import tensorflow as tf
import tensorflow_addons as tfa
from dipgnn.utils.register import registers
from dipgnn.utils.common_util import LinearWarmupExponentialDecay


@registers.task.register("base_classification_trainer")
class BaseClassificationTrainer:
    """
    Note: A significant portion of the code is adapted from dimenet (https://github.com/gasteigerjo/dimenet).
    """
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        warmup_steps=None,
        decay_steps=100000,
        decay_rate=0.96,
        ema_decay=0.999,
        max_grad_norm=10.0
    ):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

        if warmup_steps is not None:
            self.learning_rate = LinearWarmupExponentialDecay(
                learning_rate, warmup_steps, decay_steps, decay_rate)
        else:
            self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)

        opt = tf.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)
        self.optimizer = tfa.optimizers.MovingAverage(opt, average_decay=self.ema_decay)

        # Initialize backup variables
        if model.built:
            self.backup_vars = [tf.Variable(var, dtype=var.dtype, trainable=False)
                                for var in self.model.trainable_weights]
        else:
            self.backup_vars = None

        self.acc_func = tf.keras.metrics.Accuracy()
        self.recall_func = tf.keras.metrics.Recall()
        self.precision_func = tf.keras.metrics.Precision()
        self.auc_func = tf.keras.metrics.AUC()

    def update_weights(self, loss, gradient_tape):
        grads = gradient_tape.gradient(loss, self.model.trainable_weights)

        global_norm = tf.linalg.global_norm(grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    def load_averaged_variables(self):
        self.optimizer.assign_average_vars(self.model.trainable_weights)

    def save_variable_backups(self):
        if self.backup_vars is None:
            self.backup_vars = [
                tf.Variable(var, dtype=var.dtype, trainable=False)
                for var in self.model.trainable_weights]
        else:
            for var, bck in zip(self.model.trainable_weights, self.backup_vars):
                bck.assign(var)

    def restore_variable_backups(self):
        for var, bck in zip(self.model.trainable_weights, self.backup_vars):
            var.assign(bck)

    @tf.function
    def train_on_batch(self, dataset_iter, metrics, use_sigmoid=False):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            raw_preds = self.model(inputs, validation_test=False, training=True)
            preds = tf.nn.sigmoid(raw_preds) if use_sigmoid else tf.nn.softmax(raw_preds)

            self.acc_func.update_state(targets[:, 0], preds[:, 0])
            self.recall_func.update_state(targets[:, 0], preds[:, 0])
            self.precision_func.update_state(targets[:, 0], preds[:, 0])
            self.auc_func.update_state(targets[:, 0], preds[:, 0])

            acc = self.acc_func.result()
            recall = self.recall_func.result()
            precision = self.precision_func.result()
            auc = self.auc_func.result()
            f1 = 2 * precision * recall / (precision + recall)

            self.acc_func.reset_states()
            self.recall_func.reset_states()
            self.precision_func.reset_states()
            self.auc_func.reset_states()

            if use_sigmoid:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets, raw_preds))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(targets, raw_preds))

        self.update_weights(loss, tape)
        metrics.update_state(loss, acc, recall, precision, auc, f1, 1)
        return loss, targets, preds

    @tf.function
    def test_on_batch(self, dataset_iter, metrics, use_sigmoid=False):
        inputs, targets = next(dataset_iter)

        raw_preds = self.model(inputs, validation_test=True, training=False)
        preds = tf.nn.sigmoid(raw_preds) if use_sigmoid else tf.nn.softmax(raw_preds)

        self.acc_func.update_state(targets[:, 0], preds[:, 0])
        self.recall_func.update_state(targets[:, 0], preds[:, 0])
        self.precision_func.update_state(targets[:, 0], preds[:, 0])
        self.auc_func.update_state(targets[:, 0], preds[:, 0])

        acc = self.acc_func.result()
        recall = self.recall_func.result()
        precision = self.precision_func.result()
        auc = self.auc_func.result()
        f1 = 2 * precision * recall / (precision + recall)

        self.acc_func.reset_states()
        self.recall_func.reset_states()
        self.precision_func.reset_states()
        self.auc_func.reset_states()

        if use_sigmoid:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets, raw_preds))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(targets, raw_preds))

        metrics.update_state(loss, acc, recall, precision, auc, f1, 1)
        return loss, targets, preds
