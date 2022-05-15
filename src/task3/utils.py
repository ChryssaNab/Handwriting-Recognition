"""
Custom Training utilities for IAM
"""

import tensorflow as tf
from typing import List


def pad_label(label: bytes, width: int = 100, pad_char: bytes = b" ") -> bytes:
    """
    Pad label of type byte string to width using pading character.
    Padding is only on the right side.

    :param label: byte string to pad
    :param width: length of padded label
    :param pad_char: character used for padding
    :return: padded byte string
    """
    to_pad = width - len(label)
    if to_pad <= 0:
        return label
    return label + pad_char * to_pad


def label_encoding(label: bytes, chars: List[str]) -> tf.Tensor:
    """
    Encode characters in a byte string label as int.
    Uses indices in chars for encoding.

    :param label: byte string to encode
    :param chars: (sorted) list of unique characters
    :return: tensor of type int containing an encoded label
    """
    s_label = label.decode()
    e_label = [chars.index(c) for c in s_label]
    t_label = tf.constant(e_label)
    return t_label


def label_decoding(label: tf.Tensor, chars: List[str]) -> tf.string:
    """
    Decode a tensor containing an encoded label.
    Uses int values in label tensor to find characters in chars.

    :param label: tensor of encoded characters
    :param chars: (sorted) list of unique characters
    :return: tf.string tensor containing decoded label
    """
    e_label = [chars[i] for i in label]
    s_label = "".join(e_label)
    return s_label


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
                tokens: List[str],
                n_epochs: int,
                optimizer: tf.keras.optimizers.Optimizer,
                metrics: List[tf.keras.metrics.Metric],
                mean_loss: tf.keras.metrics.Metric = tf.keras.metrics.Mean(),
                batch_size: int = 32,
                ) -> List[tf.keras.metrics.Metric]:
    """
    Custom training loop for sequential models with CTC loss.

    :param model: sequential model
    :param dataset: input images and byte string labels
    :param tokens: (sorted) list of unique characters
    :param n_epochs: training epochs
    :param optimizer: model optimizer
    :param metrics: CER, WER, ...
    :param mean_loss: metric to calculate mean
    :param batch_size: batch size of training batches
    :return: training metrics
    """

    # for status bar
    n_samples = dataset.cardinality() - dataset.cardinality() % batch_size
    # input width for CTC loss
    logit_length = tf.constant(model.get_layer(index=0).input.shape[1], shape=(batch_size))
    # create new batches for each epoch
    unbatched_dataset = dataset.shuffle(batch_size * 10)

    for epoch in range(1, n_epochs + 1):
        dataset = unbatched_dataset.batch(batch_size=batch_size, drop_remainder=True)
        for step, batch in dataset.enumerate().as_numpy_iterator():
            X_batch, y_batch = batch[0], batch[1]
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)

                # get label lengths and encodings for CTC loss
                label_length = tf.constant(list(map(lambda y: len(y), y_batch)))
                labels = list(map(lambda y: pad_label(y), y_batch))
                labels = list(map(lambda y: label_encoding(y, tokens), labels))
                labels = tf.convert_to_tensor(labels)

                # calculate loss
                main_loss = tf.nn.ctc_loss(labels, y_pred, label_length, logit_length, logits_time_major=False)
                loss = tf.add_n([main_loss] + model.losses)

            # update weights
            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            mean_loss(loss)

            # calculate metrics
            for metric in metrics:
                metric(y_batch, y_pred)

            # show metrics
            print_status_bar(step * batch_size, n_samples, mean_loss, metrics)
        print_status_bar(n_samples, n_samples, mean_loss, metrics)

        # reset metrics after each epoch
        for metric in [mean_loss] + metrics:
            metric.reset_states()

    return metrics
