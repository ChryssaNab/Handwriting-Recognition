"""
Train and test a model on the IAM dataset.
"""

import tensorflow as tf
from pathlib import Path
from jiwer import wer, cer
from tqdm import tqdm

from settings import get_lstm_settings, save_settings, load_settings
from data import load_data_dict, load_dataset, split_data_dict, train_test_split, filter_labels
from data import tokens_from_text, to_dict, get_iam_token_set
from preprocessing import LabelEncoder, LabelPadding, preprocess_train_data, preprocess_test_data
from model import build_lstm_model, remove_ctc_loss_layer, CTCDecodingLayer
from metrics import ErrorRateCallback
from utils import get_parser, make_dirs, track_time, show_sample

# Debugging
DEBUG = False

# Set path to the IAM folder for training (or pass as arg)
LOCAL_PATH_TO_IAM = None


def train_model(final: bool = False) -> None:
    """
    Train the model with the settings provided in 'settings.py'.

    :param final: if set to 'True', train on the whole dataset
    """
    global DEBUG, LOCAL_PATH_TO_IAM

    # IAM data
    data_dir = Path(LOCAL_PATH_TO_IAM)
    if data_dir.name != "IAM-data":
        data_dir = data_dir / "IAM-data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Path {str(data_dir)} does not exist.")
    img_dir = data_dir / "img"

    # Create paths for logs, models, checkpoints
    print("Setting up results folder...")
    root_dir = Path("./") / "results"
    paths = make_dirs(root_dir)

    # Load & save settings
    s = get_lstm_settings(debug=bool(DEBUG))
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
    dataset = load_dataset(data_dict, img_dir, return_filenames=True)

    # Get tokens
    full_text = "".join(data_dict.values())
    tokens = tokens_from_text(full_text)
    pad_value = len(tokens) + 3
    ctc_blank = s["ctc_blank"]

    # Prepare label encoding & padding
    label_encoder = LabelEncoder(tokens)
    label_padding = LabelPadding(pad_value=pad_value, max_len=max_label_len)

    # preprocess data
    dataset = preprocess_train_data(dataset, label_encoder, label_padding, image_width, image_height)

    # Change to dict format, prepare for input
    dataset = dataset.map(lambda x, y, f: to_dict(x, y, f, f_key="Filename"))
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
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=paths.checkpoint / f"{model_name}_checkpoint.h5",
                                                       save_weights_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    callbacks = [error_cb, tensorboard_cb, checkpoint_cb, early_stopping_cb]

    # Use only single batches for debugging
    if s["debug"]:
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)
        test_ds = test_ds.take(1)
        dataset = dataset.take(1)

    # Use whole dataset for training final model, no validation data & no early stopping
    if final:
        train_ds = dataset
        val_ds = None
        epochs = 25
        callbacks.remove(early_stopping_cb)
        callbacks.remove(error_cb)

    # Train
    history = train_model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks,
                              )

    # Remove loss layer after training
    final_model = remove_ctc_loss_layer(train_model, model_name)

    # Save model
    final_model.save(paths.model / f"{model_name}.h5")

    # Skip test when finalizing model
    if final:
        return

    # Test
    batch = test_ds
    wer_hist, cer_hist = [], []
    y_pred = final_model.predict(test_ds.map(filter_labels))
    y_pred = tf.convert_to_tensor(y_pred, dtype="int32")
    for i, sample in tqdm(iter(batch.unbatch().enumerate())):
        y_true, filename = sample["Label"], sample["Filename"]
        filename = bytes(filename.numpy()).decode()
        y_true = label_padding.remove(y_true)
        y_true = label_encoder.decode(y_true)

        output = y_pred[i]
        output = label_padding.remove(output, pad_value=ctc_blank)
        output = label_encoder.decode(output)

        wer_score = wer(y_true, output)
        cer_score = cer(y_true, output)
        wer_hist.append(wer_score)
        cer_hist.append(cer_score)

        with open(paths.logs / "output.txt", "a") as f:
            f.write(f"File: {filename}\n")
            f.write(f"y_true: {y_true}\n")
            f.write(f"y_pred: {output}\n")
            f.write(f"WER: {wer_score:.4f}\n")
            f.write(f"CER: {cer_score:.4f}\n\n")

    with open(paths.logs / "output.txt", "a") as f:
        f.write(f"Mean WER: {tf.reduce_mean(wer_hist):.4f}")
        f.write(f"Mean CER: {tf.reduce_mean(cer_hist):.4f}")

    print(f"Mean WER: {tf.reduce_mean(wer_hist):.4f}")
    print(f"Mean CER: {tf.reduce_mean(cer_hist):.4f}")


def test_model(model_path, img_path) -> None:
    """
    Test a trained model on new data and save results.
    Each image generates one txt file.

    :param model_path: path to trained model
    :param img_path: path to IAM img folder
    """

    global DEBUG

    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects={"CTCDecodingLayer": CTCDecodingLayer})
    print(model.summary())

    # Load model settings
    model_settings = load_settings(Path("./results/run_final_model/settings/LSTM_model_settings.json"))
    image_width = model_settings["image_width"]

    # Load & preprocess images
    filenames = list(img_path.glob("*.png"))
    filenames = [str(f.name) for f in filenames]
    data_dict = dict.fromkeys(filenames, "N/A")
    dataset = load_dataset(data_dict, img_dir=img_path, return_filenames=True)
    dataset = dataset.map(lambda x, y, f: (x, f))

    # Preprocess data
    preprocess = lambda d: preprocess_test_data(d, image_width=image_width)
    dataset = dataset.apply(preprocess).batch(1)

    # Get tokens
    tokens = get_iam_token_set()
    pad_value = len(tokens) + 3
    ctc_blank = model_settings["ctc_blank"]

    # Prepare label encoding & padding
    max_label_len = 99
    label_encoder = LabelEncoder(tokens)
    label_padding = LabelPadding(pad_value=pad_value, max_len=max_label_len)

    # Create results folder
    results_path = Path("./results")
    results_path.mkdir(exist_ok=True)

    if DEBUG:
        dataset = dataset.take(20)

    # Run model & save output
    for img, filename in tqdm(iter(dataset)):
        y_pred = model.predict({"Image": img})
        output = y_pred[0]
        f_name = bytes(filename.numpy()[0]).decode()
        output = label_padding.remove(output, pad_value=ctc_blank)
        output = label_encoder.decode(output)

        with open(results_path / f_name.replace(".png", ".txt"), "w") as f:
            f.write(output)


@track_time
def main():
    """
    Train or test model in standard or debug mode.
    """
    global DEBUG, LOCAL_PATH_TO_IAM

    # CMD args
    parser = get_parser()
    args = parser.parse_args()
    mode = args.mode
    DEBUG = args.debug

    # Data & model path
    img_path = args.path
    model_path = Path("./results/run_final_model/model/LSTM_model.h5")

    # Run train or test
    if mode == "train":
        if img_path is not None:
            LOCAL_PATH_TO_IAM = img_path
        train_model(final=args.final)
    elif mode == "test":
        if img_path is None:
            raise IOError("Path to IAM images is missing.")
        test_model(model_path=model_path, img_path=Path(img_path))


if __name__ == "__main__":
    main()
