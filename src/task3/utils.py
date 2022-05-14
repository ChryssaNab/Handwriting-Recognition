"""
Custom Training utilities for IAM
"""

import tensorflow as tf
from typing import List


def pad_label(label: bytes, width: int = 100, pad_char: bytes = b" ") -> bytes:
    to_pad = width - len(label)
    if to_pad <= 0:
        return label
    return label + pad_char * to_pad


def print_status_bar(iteration: int, total: int, loss, metrics=None):
    """
    Custom function to print training progress and metrics

    :param iteration: current step in epoch
    :param total: number of steps per epoch
    :param loss: (mean) loss since start of epoch
    :param metrics: other metrics (CER, WER)
    """
    metrics = " - ".join([f"{m.name}: {m.result():.4f}" for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print(f"\r{iteration:<5}/{total:>5}\t{metrics}", end=end)


def train_model(model: tf.keras.Sequential,
                dataset: tf.data.Dataset,
                n_epochs: int,
                loss_fn,
                optimizer: tf.keras.optimizers.Optimizer,
                metrics: List[tf.keras.metrics.Metric],
                mean_loss: tf.keras.metrics.Metric = tf.keras.metrics.Mean(),
                batch_size: int = 32,
                ):

    # for status bar
    n_samples = dataset.cardinality() - dataset.cardinality() % batch_size
    # input width for CTC loss
    logit_length = tf.constant(model.get_layer(index=0).input.shape[1], shape=(batch_size))
    # create new batches for each epoch
    unbatched_dataset = dataset.shuffle(dataset.cardinality())
    for epoch in range(1, n_epochs + 1):
        dataset = unbatched_dataset.batch(batch_size=batch_size, drop_remainder=True)
        for step, batch in dataset.enumerate().as_numpy_iterator():
            X_batch, y_batch = batch[0], batch[1]
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                label_length = list(map(lambda x: len(x), y_batch))
                labels = tf.constant(list(map(lambda x: pad_label(x), y_batch)))
                main_loss = tf.nn.ctc_loss(labels, y_pred, label_length, logit_length, logits_time_major=False)
                loss = tf.add_n([main_loss] + model.losses)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            mean_loss(loss)

            for metric in metrics:
                metric(y_batch, y_pred)

            print_status_bar(step * batch_size, n_samples, mean_loss, metrics)
        print_status_bar(n_samples, n_samples, mean_loss, metrics)

        for metric in [mean_loss] + metrics:
            metric.reset_states()
