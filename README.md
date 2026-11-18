# Détection de défauts sur surfaces métalliques (NEU)

Classification par CNN (TensorFlow / Keras) des six défauts du jeu de données **NEU Surface Defect Database**
(crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) avec une interface **Streamlit**.

## Répartition du travail

| Personne | Partie | Livrables |
|---|---|---|
| 1 | Data & modèle IA | notebooks 01 et 02, `models/best_model.keras`, `results/accuracy.png`, `results/loss.png` |
| 2 | Évaluation & application | notebook 03, `results/confusion_matrix.png`, `app/streamlit_app.py`, captures d'écran |

## Structure

```
metal-defect-detection/
├── notebooks/
│   ├── 01_exploration_preprocessing.ipynb   # P1
│   ├── 02_cnn_training.ipynb                # P1
│   └── 03_model_evaluation.ipynb            # P2
├── models/best_model.keras                  # P1
├── app/streamlit_app.py                     # P2
├── src/                                     # code partagé (chemins, modèle, découpage des données)
├── results/                                 # courbes, matrice de confusion, rapports
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Données

1. Télécharger **NEU Surface Defect Database** (Kaggle ou site officiel) et le décompresser dans `data_raw/`.
2. Créer le découpage train / validation / test (70 / 15 / 15, graine fixe, identique pour tous) :

```bash
python src/prepare_dataset.py --raw data_raw
```

## Exécution

```bash
jupyter lab                                  # ouvrir les notebooks 01 -> 02 -> 03
streamlit run app/streamlit_app.py           # lancer l'application
```

Sans modèle entraîné, `python src/make_dummy_model.py` crée un modèle factice pour tester l'interface.

## Résultats

À compléter par l'équipe : accuracy, F1-score, matrice de confusion (`results/`), captures de l'application.
