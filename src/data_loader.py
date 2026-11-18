"""Chargement des données avec TensorFlow / Keras (images en niveaux de gris 200x200)."""
import tensorflow as tf
from tensorflow import keras

from src.common import CLASS_NAMES, DATASET_DIR, IMG_SIZE, SEED


def load_split(subset: str, batch_size: int = 32, shuffle: bool = False,
               return_paths: bool = False):
    """Charge dataset/<subset>/ ('train', 'validation' ou 'test').

    Les labels sont des entiers 0..5 dans l'ordre de CLASS_NAMES.
    Pour l'évaluation, garder shuffle=False afin que l'ordre des prédictions
    corresponde à celui des fichiers (return_paths=True).
    """
    ds = keras.utils.image_dataset_from_directory(
        DATASET_DIR / subset,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
    )
    paths = list(ds.file_paths)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return (ds, paths) if return_paths else ds


def load_all(batch_size: int = 32):
    """Retourne (train, validation, test)."""
    return (
        load_split("train", batch_size, shuffle=True),
        load_split("validation", batch_size),
        load_split("test", batch_size),
    )
