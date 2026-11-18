"""Application Streamlit : détection de défauts sur surfaces métalliques (NEU).

Lancer depuis la racine du projet :
    streamlit run app/streamlit_app.py
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    BEST_MODEL_PATH,
    CLASS_LABELS_FR,
    CLASS_NAMES,
    DATASET_DIR,
    DUMMY_MODEL_PATH,
    IMAGE_EXTENSIONS,
    preprocess_pil,
)

LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Défauts de surface - NEU", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(path: str):
    from tensorflow import keras
    return keras.models.load_model(path)


def find_model():
    """Retourne (chemin, est_factice)."""
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH, False
    if DUMMY_MODEL_PATH.exists():
        return DUMMY_MODEL_PATH, True
    return None, False


def list_test_samples():
    test_dir = DATASET_DIR / "test"
    if not test_dir.exists():
        return []
    return sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


# --------------------------------------------------------------------------- #
# En-tête et barre latérale
# --------------------------------------------------------------------------- #
st.title("🔍 Détection de défauts sur surfaces métalliques")
st.caption("Classification par CNN des 6 types de défauts du jeu de données NEU Surface Defect Database.")

model_path, is_dummy = find_model()

with st.sidebar:
    st.header("À propos")
    st.write("Importez une image de surface métallique (acier laminé à chaud) "
             "pour identifier le défaut présent.")
    st.subheader("Les 6 catégories")
    for name in CLASS_NAMES:
        st.markdown(f"- {CLASS_LABELS_FR[name]}")
    st.divider()
    source = st.radio("Source de l'image", ["Importer un fichier", "Exemple du jeu de test"])

if model_path is None:
    st.error("Aucun modèle trouvé dans `models/`. Placez `best_model.keras` (fourni par la "
             "Personne 1) ou générez un modèle factice : `python src/make_dummy_model.py`.")
    st.stop()

if is_dummy:
    st.warning("⚠️ Modèle FACTICE non entraîné : les prédictions sont aléatoires. "
               "Placez `models/best_model.keras` pour obtenir de vrais résultats.")

model = load_model(str(model_path))

# --------------------------------------------------------------------------- #
# Choix de l'image
# --------------------------------------------------------------------------- #
image, true_label, caption = None, None, ""

if source == "Importer un fichier":
    uploaded = st.file_uploader("Importer une image (BMP, PNG, JPG)",
                                type=["bmp", "png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            image.load()
            caption = uploaded.name
        except Exception:
            st.error("Fichier illisible : importez une image valide (BMP, PNG ou JPG).")
else:
    samples = list_test_samples()
    if not samples:
        st.info("Dossier `dataset/test/` absent : lancez `python src/prepare_dataset.py --raw data_raw`.")
    else:
        if "sample_idx" not in st.session_state:
            st.session_state.sample_idx = random.randrange(len(samples))
        if st.button("🎲 Image aléatoire"):
            st.session_state.sample_idx = random.randrange(len(samples))
        path = samples[min(st.session_state.sample_idx, len(samples) - 1)]
        image = Image.open(path)
        true_label = path.parent.name
        caption = path.name

# --------------------------------------------------------------------------- #
# Prédiction et affichage
# --------------------------------------------------------------------------- #
if image is None:
    st.info("👆 Choisissez une image pour lancer l'analyse.")
    st.stop()

x = preprocess_pil(image)
probs = model.predict(x, verbose=0)[0]
best = int(np.argmax(probs))
pred_name, confidence = CLASS_NAMES[best], float(probs[best])

col_img, col_res = st.columns([1, 1.3], gap="large")

with col_img:
    st.subheader("Image analysée")
    st.image(image.convert("L"), caption=caption, width=420)
    st.caption(f"Prétraitement : niveaux de gris, redimensionnée en {x.shape[2]}×{x.shape[1]} px.")

with col_res:
    st.subheader("Résultat")
    st.success(f"Défaut prédit : **{CLASS_LABELS_FR[pred_name]}**")
    st.metric("Niveau de confiance", f"{confidence:.1%}")
    st.progress(min(max(confidence, 0.0), 1.0))
    if confidence < LOW_CONFIDENCE:
        st.warning("Confiance faible : le résultat est incertain, à vérifier manuellement.")
    if true_label is not None:
        if true_label == pred_name:
            st.info(f"✅ Classe réelle : {true_label} - prédiction correcte.")
        else:
            st.error(f"❌ Classe réelle : {true_label} - prédiction incorrecte.")

    st.subheader("Probabilités des six classes")
    df = pd.DataFrame({"Classe": CLASS_NAMES, "Probabilité": probs}).set_index("Classe")
    st.bar_chart(df, horizontal=True)
    table = df.sort_values("Probabilité", ascending=False).copy()
    table["Probabilité"] = table["Probabilité"].map(lambda p: f"{p:.2%}")
    st.dataframe(table)
