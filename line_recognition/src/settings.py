"""
Settings for IAM main
"""

import json
from pathlib import Path
from typing import Dict, Any


def get_lstm_settings(debug: bool = False) -> Dict[str, Any]:
    """
    Settings for the LSTM model.

    :param debug: use debug settings
    :return: dict with settings
    """

    settings = {

        # Training
        "epochs":           30,
        "batch_size":       24,
        "optimizer":        {
                             'class_name':  'RMSprop',
                             'config':      {
                                             'name':            'RMSprop',
                                             'learning_rate':   0.00025,
                                             'decay':           0.0,
                                             'rho':             0.9,
                                             'momentum':        0.0,
                                             'epsilon':         1e-07,
                                             'centered':        False,
                                            },
                            },

        # Image size
        "image_width":      1000,
        "image_height":     64,

        # Label length
        "max_label_length": 80,

        # Dataset
        "test_split":       0.85,
        "validation_split": 0.9,

        # CTC
        "ctc_blank":        (-1),

        # ID
        "model_name":       "LSTM_model",

        # Debug mode
        "debug":            False,
    }

    if debug:
        settings["epochs"] = 2
        settings["model_name"] += "_debug"
        settings["debug"] = True

    return settings


def save_settings(settings: dict, path: Path, filename: str = None) -> None:
    """
    Save settings dict as json file.

    :param settings: settings as dict
    :param path: save location
    :param filename: name of settings file
    """
    if filename is None:
        filename = f"{settings['model_name']}_settings.json"

    with open(str(path / filename), "w") as f:
        json.dump(settings, f)


def load_settings(path: Path) -> Dict[str, Any]:
    """
    Load settings as dict from json file.

    :param path: path to json file
    :return: settings as dict
    """
    with open(str(path), "r") as f:
        settings = json.load(f)

    if not isinstance(settings, dict):
        raise TypeError("Cannot load settings as dict")

    return settings
