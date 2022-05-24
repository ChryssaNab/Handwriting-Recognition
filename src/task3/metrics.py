"""
Metrics for IAM
"""

import tensorflow as tf
from jiwer import wer, cer

from preprocessing import LabelEncoder, LabelPadding


# TODO: log output since its not a metric.
class ErrorRateCallback(tf.keras.callbacks.Callback):
    """
    Calculates and prints WER & CER for validation dataset each epoch.
    Implemented as callback instead of metric for simplicity.
    """

    def __init__(self,
                 val_ds: tf.data.Dataset,
                 label_encoder: LabelEncoder,
                 label_padding: LabelPadding,
                 ctc_blank: int = -1) -> None:
        """
        Needs extra utilities to remove padding and decode labels.

        :param val_ds: validation data
        :param label_encoder: to decode labels
        :param label_padding: to remove padding from labels
        :param ctc_blank: padding character to remove
        """

        super(ErrorRateCallback, self).__init__()
        self.val_ds = val_ds
        self.label_encoder = label_encoder
        self.label_padding = label_padding
        self.ctc_blank = ctc_blank

    def on_epoch_end(self, epoch, logs=None) -> None:
        wer_epoch = []
        cer_epoch = []

        for batch in iter(self.val_ds):
            y_pred = self.model.predict(batch)
            y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int64)

            for i, (img, y_true) in enumerate(zip(batch["Image"], batch["Label"])):
                y_true = self.label_padding.remove(y_true)
                y_true = self.label_encoder.decode(y_true)
                output = y_pred[i]
                output = self.label_padding.remove(output, pad_value=self.ctc_blank)
                output = self.label_encoder.decode(output)
                wer_epoch.append(wer(y_true, output))
                cer_epoch.append(cer(y_true, output))

        mean_wer = tf.reduce_mean(wer_epoch)
        mean_cer = tf.reduce_mean(cer_epoch)

        print(
            f"WER for epoch {epoch + 1}: {mean_wer:.4f}\n"
            f"CER for epoch {epoch + 1}: {mean_cer:.4f}"
        )
