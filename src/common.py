"""Constantes et fonctions partagées par Personne 1 et Personne 2.

Ce module n'importe PAS TensorFlow : il peut donc être utilisé partout
(notebooks, script de préparation, application Streamlit).
"""
from pathlib import Path

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# Chemins (toujours relatifs à la racine du projet -> fonctionne sur tout PC)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data_raw"
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
DUMMY_MODEL_PATH = MODELS_DIR / "dummy_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# --------------------------------------------------------------------------- #
# Paramètres du problème
# --------------------------------------------------------------------------- #
IMG_SIZE = (200, 200)          # (hauteur, largeur) - taille native de NEU
INPUT_SHAPE = (200, 200, 1)    # images en niveaux de gris
SEED = 42

# ORDRE FIXE : c'est l'ordre des sorties du modèle (indice 0 à 5).
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_LABELS_FR = {
    "crazing": "Crazing (fissures superficielles)",
    "inclusion": "Inclusion (inclusions)",
    "patches": "Patches (zones irrégulières)",
    "pitted_surface": "Pitted Surface (surface piquée)",
    "rolled-in_scale": "Rolled-in Scale (écailles d'oxydation)",
    "scratches": "Scratches (rayures)",
}

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}


# --------------------------------------------------------------------------- #
# Prétraitement d'une image pour le modèle
# --------------------------------------------------------------------------- #
def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en tenseur (1, 200, 200, 1) pour le modèle.

    - niveaux de gris, redimensionnement 200x200 (interpolation bilinéaire,
      comme image_dataset_from_directory) ;
    - les valeurs restent en 0-255 : la normalisation (÷255) est faite par la
      première couche (Rescaling) DU MODÈLE, ce qui évite tout décalage entre
      l'entraînement et l'application.
    """
    gray = img.convert("L").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    arr = np.asarray(gray, dtype="float32")
    return arr[np.newaxis, :, :, np.newaxis]
