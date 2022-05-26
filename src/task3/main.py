"""
Train and test a model on the IAM dataset.
"""

################################################## DEBUGGING ###########################################################

GPU_OFF = 0

if GPU_OFF:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

########################################################################################################################

import tensorflow as tf

from pathlib import Path
from jiwer import wer, cer

from settings import get_lstm_settings, save_settings, load_settings
from data import load_data_dict, load_dataset, split_data_dict, train_test_split, filter_labels, to_dict, from_dict
from data import tokens_from_text, get_full_token_set
from preprocessing import invert_color, distortion_free_resize, scale_img
from preprocessing import LabelEncoder, LabelPadding
from model import build_lstm_model, remove_ctc_loss_layer
from metrics import ErrorRateCallback
from utils import get_parser, make_dirs, track_time

# Set path to the IAM folder
local_path_to_iam = "C:\\Users\\Luca\\Desktop\\HWR"
#local_path_to_iam = "C:\\Users\\muell\\Desktop\\HWR\\Task 3\\Data"


@track_time
def main():
    global local_path_to_iam

    # cmd args
    parser = get_parser()
    args = parser.parse_args()

    # TODO: make separate train / test loop
    mode = args.mode
    debug = args.debug

    if args.path is not None:
        local_path_to_iam = args.path

    # IAM data
    data_dir = Path(local_path_to_iam)
    if data_dir.name != "IAM-data":
        data_dir = data_dir / "IAM-data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Path {str(data_dir)} does not exist.")
    img_dir = data_dir / "img"

    # Create paths for logs, models, checkpoints
    print("Setting up results folder...")
    root_dir = Path("../") / "iam_results"
    paths = make_dirs(root_dir)

    # Load & save settings
    s = get_lstm_settings(debug=bool(debug))
    save_settings(s, paths.settings)

    # Model settings
    epochs = s["epochs"]
    batch_size = s["batch_size"]
    optimizer = tf.keras.optimizers.get(s["optimizer"])
    model_name = s["model_name"]

    # Set input dimensions
    image_width = s["image_width"]
    image_height = s["image_height"]

    # longest label
    max_label_len = s["max_label_length"]

    # Load data
    data_dict = load_data_dict(data_dir)
    dataset = load_dataset(data_dict, img_dir)

    # Get tokens
    full_text = "".join(data_dict.values())
    tokens = tokens_from_text(full_text)
    pad_value = len(tokens) + 3
    ctc_blank = s["ctc_blank"]

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
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Split data: Train = 0.765, Valid = 0.085, Test = 0.15
    train_ds, test_ds = train_test_split(dataset, train_size=s["test_split"], shuffle=False)
    train_ds, val_ds = train_test_split(train_ds, train_size=s["validation_split"], shuffle=True)

    # Model for training
    train_model = build_lstm_model(len(tokens), image_width)
    train_model.compile(optimizer=optimizer)
    print(train_model.summary())

    # Callbacks
    error_cb = ErrorRateCallback(val_ds, label_encoder, label_padding, log_dir=paths.logs / "validation")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=paths.logs)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(paths.checkpoint / f"{model_name}_checkpoint.h5")
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callbacks = [error_cb, tensorboard_cb, checkpoint_cb, early_stopping_cb]

    # Use only single batches for debugging
    if s["debug"]:
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)
        test_ds = test_ds.take(1)

    # Train
    history = train_model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks,
                              )

    # Remove loss layer after training
    final_model = remove_ctc_loss_layer(train_model, model_name)

    # Test
    batch = val_ds.take(1)
    y_pred = final_model.predict(val_ds.map(filter_labels))
    y_pred = tf.convert_to_tensor(y_pred, dtype="int32")
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
        cer_score = cer(y_true, output)
        print(f"wer: {wer_score:.4f}")
        print(f"cer: {cer_score:.4f}\n")

    # Save model
    final_model.save(paths.model / f"{model_name}.h5")


if __name__ == "__main__":
    main()
