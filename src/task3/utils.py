"""
Custom Training utilities for IAM
"""

# TODO: put functions in proper modules
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List


def show_sample(X: tf.Tensor, y: tf.Tensor):
    plt.imshow(tf.transpose(tf.image.flip_left_right(X), [1, 0, 2]), cmap='Greys')
    plt.title(y)
    plt.show()


def label_encoding(label: bytes, tokens: List[str]) -> tf.Tensor:
    """
    Encode characters in a byte string label as int.
    Uses indices in tokens for encoding.

    :param label: byte string to encode
    :param tokens: (sorted) list of unique characters
    :return: tensor of type int containing an encoded label
    """
    s_label = label.decode()
    e_label = [tokens.index(c) for c in s_label]
    t_label = tf.constant(e_label)
    return t_label


def label_decoding(label: tf.Tensor, tokens: List[str]) -> str:
    """
    Decode a tensor containing an encoded label.
    Uses int values in label tensor to find characters in tokens.

    :param label: tensor of encoded characters
    :param tokens: (sorted) list of unique characters
    :return: tf.string tensor containing decoded label
    """
    e_label = [tokens[i] for i in label]
    s_label = "".join(e_label)
    return str(s_label)


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
    print(f"\r{iteration:>5}/{total:<5}\t{metrics}", end=end)


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

    # for CTC
    tokens.append("ε")
    # for status bar
    n_samples = dataset.cardinality() - dataset.cardinality() % batch_size
    # T for CTC loss
    logit_length = tf.constant(200, shape=batch_size)
    # create new batches for each epoch
    unbatched_dataset = dataset.shuffle(buffer_size=batch_size * 10, reshuffle_each_iteration=True)

    # TODO: speed up training
    @tf.function
    def train_step(x, labels: tf.Tensor, label_length: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            logits = tf.transpose(logits, [1, 0, 2])
            loss_value = tf.nn.ctc_loss(labels, logits, label_length, logit_length, logits_time_major=True)
            loss = tf.add_n([loss_value] + model.losses)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        mean_loss(loss)
        return loss_value

    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch:>3}/{n_epochs:<3}")
        dataset = unbatched_dataset.batch(batch_size=batch_size, drop_remainder=True)

        for step, batch in dataset.enumerate().as_numpy_iterator():
            (X_batch, y_batch) = batch

            # get label lengths and encodings for CTC loss
            label_length = tf.constant([len(y) for y in y_batch])
            labels = list(map(lambda y: label_encoding(y, tokens), y_batch))
            labels = tf.keras.preprocessing.sequence.pad_sequences(labels, len(tokens), padding='post')
            labels = tf.convert_to_tensor(labels)

            # TODO: find out why blank probs explode
            with tf.GradientTape() as tape:
                # get predictions and transpose to get time major representation
                y_pred = model(X_batch, training=True)
                y_pred = tf.transpose(y_pred, [1, 0, 2])

                #y = y_pred[80, 0]
                #print(f"\tmax: {np.argmax(y, axis=-1)}, min: {np.argmin(y, axis=-1)}, shape: {y.shape}")

                # calculate loss
                loss = tf.nn.ctc_loss(labels, y_pred, label_length, logit_length, blank_index=len(tokens) - 1)

            # update weights
            gradients = tape.gradient(loss, model.trainable_weights)
            #for g in gradients:
            #    print(f"max: {tf.math.reduce_max(g, axis=-1)}, min: {tf.reduce_min(g, axis=-1)}")
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


# TODO: Finish this function
def test_model(model: tf.keras.Sequential,
               X: tf.data.Dataset,
               seq_lens: tf.Tensor,
               tokens: List[str],
               metrics: List[tf.keras.metrics.Metric] = None,
               mean_loss: tf.keras.metrics.Metric = tf.keras.metrics.Mean(),
               ):

    y_decoded = list()

    for batch in iter(X):
        if isinstance(X, tf.data.Dataset):
            y_pred = model.predict(batch)
        else:
            y_pred = model(X, training=False)
        y_pred = tf.transpose(y_pred, [1, 0, 2])

        (decoded, _) = tf.nn.ctc_beam_search_decoder(y_pred, seq_lens)
        decoded = tf.sparse.to_dense(decoded[0])
        for sample in decoded:
            y_decoded.append(str(label_decoding(sample, tokens)))

    return y_decoded
