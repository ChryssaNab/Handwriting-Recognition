"""
Train and test a model on the IAM dataset.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import tensorflow as tf

from pathlib import Path
from jiwer import wer

from task3.data import load_data_dict, load_dataset, train_test_split, to_dict, from_dict
from task3.data import tokens_from_text, get_full_token_set
from task3.preprocessing import invert_color, distortion_free_resize, scale_img
from task3.preprocessing import LabelEncoder, LabelPadding
from task3.model import build_LSTM_model, build_model
from task3.metrics import ErrorRateCallback


# Set path to the IAM folder
#local_path_to_iam = "C:\\Users\\Luca\\Desktop\\HWR"
local_path_to_iam = "C:\\Users\\muell\\Desktop\\HWR\\Task 3\\Data"


def main():
    global local_path_to_iam

    if len(sys.argv) > 1:
        local_path_to_iam = str(sys.argv[1])

    data_dir = Path(local_path_to_iam) / "IAM-data"
    img_dir = data_dir / "img"
    print(f"IAM Path: {data_dir}")

    # Settings
    EPOCHS = 1
    BATCH_SIZE = 24
    LEARNING_RATE = 0.001
    OPTIMIZER = tf.keras.optimizers.RMSprop(LEARNING_RATE)
    METRICS = []

    # Set input dimensions
    image_width = 800
    image_height = 64
    image_channels = 1
    input_shape = (image_width, image_height, image_channels)

    # Load data
    data_dict = load_data_dict(data_dir)
    dataset = load_dataset(data_dict, img_dir)

    # Get tokens
    full_text = "".join(data_dict.values())
    tokens = tokens_from_text(full_text)
    pad_value = len(tokens) + 3
    ctc_blank = -1

    # longest label
    max_label_len = max(list(map(len, data_dict.values())))

    # Prepare label encoding & padding
    label_encoder = LabelEncoder(tokens)
    label_padding = LabelPadding(pad_value=pad_value, max_len=max_label_len)

    # Pre-process images
    dataset = dataset.map(lambda x, y: (invert_color(x), y))
    dataset = dataset.map(lambda x, y: (distortion_free_resize(x, img_size=(image_width, image_height),
                                                               pad_value=0), y))
    dataset = dataset.map(lambda x, y: (scale_img(x), y))

    # Pre-process labels
    dataset = dataset.map(lambda x, y: (x, label_encoder.encode(y)))
    dataset = dataset.map(lambda x, y: (x, label_padding.add(y)))

    # Change to dict format, prepare for input
    dataset = dataset.map(to_dict)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # Split data
    train_ds, test_ds = train_test_split(dataset, train_size=0.8, shuffle=True)

    train_model = build_LSTM_model(len(tokens), image_width)

    train_model.compile(optimizer=OPTIMIZER)

    softmax = train_model.get_layer(name="Output_Softmax").output
    output = train_model.get_layer(name="CTC_Decoding")(softmax)
    final_model = tf.keras.models.Model(
        inputs=train_model.get_layer(name="Image").input, outputs=output, name="LSTM_model_trained"
    )

    print(train_model.summary())
    err_callback = ErrorRateCallback(train_model, train_ds.take(10), label_encoder, label_padding)

    history = train_model.fit(train_ds.skip(100).take(1),
                              validation_data=train_ds.take(1),
                              epochs=EPOCHS,
                              callbacks=[err_callback]
                              )

    batch = train_ds.take(1)
    y_pred = final_model.predict(batch)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int64)
    for i, sample in iter(batch.unbatch().enumerate()):
        img, y_true = sample["Image"], sample["Label"]
        y_true = label_padding.remove(y_true)
        y_true = label_encoder.decode(y_true)
        print(f"y_true: {y_true}")
        output = y_pred[i]
        output = label_padding.remove(output, pad_value=ctc_blank)
        output = label_encoder.decode(output)
        print(f"y_pred: {output}")
        wer_score = wer(y_true, output)
        print(f"wer: {wer_score}")


if __name__ == "__main__":
    main()
