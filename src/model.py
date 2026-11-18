"""Architecture du CNN (Personne 1) - utilisée aussi pour le modèle factice de Personne 2."""
from tensorflow import keras
from tensorflow.keras import layers

from src.common import CLASS_NAMES, INPUT_SHAPE


def build_cnn(input_shape=INPUT_SHAPE, num_classes=len(CLASS_NAMES),
              filters=(32, 64, 128, 128), dense_units=128, dropout=0.4,
              learning_rate=1e-3, augment=True):
    """CNN simple : 4 blocs Conv-BatchNorm-ReLU-MaxPool + tête de classification.

    Entrée : image 200x200x1 en valeurs 0-255 (la normalisation est DANS le modèle).
    Sortie : 6 probabilités (softmax) dans l'ordre de CLASS_NAMES.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255, name="normalisation")(inputs)

    if augment:  # actives seulement à l'entraînement, ignorées à l'inférence
        x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(x)
        x = layers.RandomRotation(0.05, name="aug_rotation")(x)

    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilites")(x)

    model = keras.Model(inputs, outputs, name="cnn_neu")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
