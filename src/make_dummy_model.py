"""(Personne 2) Crée un modèle NON ENTRAÎNÉ pour développer l'interface Streamlit
avant que Personne 1 ait livré best_model.keras.

    python src/make_dummy_model.py

Le fichier models/dummy_model.keras est ignoré par Git. Ses prédictions sont aléatoires :
elles ne servent qu'à tester l'interface.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DUMMY_MODEL_PATH  # noqa: E402
from src.model import build_cnn  # noqa: E402

if __name__ == "__main__":
    DUMMY_MODEL_PATH.parent.mkdir(exist_ok=True)
    build_cnn().save(DUMMY_MODEL_PATH)
    print(f"Modèle factice enregistré : {DUMMY_MODEL_PATH}")
