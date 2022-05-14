"""
Custom Training utilities for IAM
"""

import tensorflow as tf
from typing import List


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
    print(f"\r{iteration}/{total}" + metrics, end=end)


def train_model(model: tf.keras.Sequential,
                dataset: tf.data.Dataset,
                n_epochs: int,
                loss_fn,
                optimizer: tf.keras.optimizers.Optimizer,
                metrics: List[tf.keras.metrics.Metric],
                mean_loss: tf.keras.metrics.Metric = tf.keras.metrics.Mean(),
                batch_size: int = 32,
                ):

    n_steps = dataset.cardinality() // batch_size
    unbatched_dataset = dataset.shuffle(dataset.cardinality())

    for epoch in range(1, n_epochs + 1):
        dataset = unbatched_dataset.batch(batch_size=batch_size, drop_remainder=True)
        for step in range(1, n_steps + 1):
            X_batch, y_batch = dataset.map(lambda x, y: x), dataset.map(lambda x, y: y)
            with tf.GradientTape as tape:
                y_pred = model(X_batch, training=True)
                # TODO: implement CTC loss
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            mean_loss(loss)

            for metric in metrics:
                metric(y_batch, y_pred)

            print_status_bar(step * batch_size, dataset.cardinality(), mean_loss, metrics)
        print_status_bar(dataset.cardinality(), dataset.cardinality(), mean_loss, metrics)

        for metric in [mean_loss] + metrics:
            metric.reset_states()
