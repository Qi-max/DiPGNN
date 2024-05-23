import tensorflow as tf
from dipgnn.utils.register import registers
from dipgnn.trainers.base_trainer import BaseTrainer


@registers.task.register("base_classification_trainer")
class BaseClassificationTrainer(BaseTrainer):
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
        max_grad_norm=10.0,
        use_sigmoid=False
    ):
        super().__init__(
            model=model,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            ema_decay=ema_decay,
            max_grad_norm=max_grad_norm,
        )

        self.use_sigmoid = use_sigmoid
        self.auc_func = tf.keras.metrics.AUC()

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            raw_preds = self.model(inputs, validation_test=False, training=True)
            preds = tf.nn.sigmoid(raw_preds) if self.use_sigmoid else tf.nn.softmax(raw_preds)

            self.auc_func.update_state(targets[:, 0], preds[:, 0])
            auc = self.auc_func.result()
            self.auc_func.reset_states()

            if self.use_sigmoid:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets, raw_preds))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(targets, raw_preds))

        self.update_weights(loss, tape)
        metrics.update_state(loss, auc, 1)
        return loss, targets, preds

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)

        raw_preds = self.model(inputs, validation_test=True, training=False)
        preds = tf.nn.sigmoid(raw_preds) if self.use_sigmoid else tf.nn.softmax(raw_preds)

        self.auc_func.update_state(targets[:, 0], preds[:, 0])
        auc = self.auc_func.result()
        self.auc_func.reset_states()

        if self.use_sigmoid:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets, raw_preds))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(targets, raw_preds))

        metrics.update_state(loss, auc, 1)
        return loss, targets, preds
