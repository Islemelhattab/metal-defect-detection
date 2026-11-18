"""Prépare dataset/{train,validation,test}/<classe>/ à partir du dataset brut NEU.

Le découpage est DÉTERMINISTE (graine fixe) : Personne 1 et Personne 2 obtiennent
exactement les mêmes ensembles train / validation / test en lançant ce script
chacun sur leur machine. Le dossier dataset/ n'est donc pas versionné sur Git.

Utilisation (depuis la racine du projet) :
    python src/prepare_dataset.py --raw data_raw
    python src/prepare_dataset.py --raw data_raw --force      # refaire le découpage

Le script accepte les deux organisations courantes du dataset :
  * dossiers par classe        : .../crazing/crazing_1.bmp
  * dossier unique, préfixes   : .../Cr_1.bmp, In_1.bmp, Pa_1.bmp, PS_1.bmp, RS_1.bmp, Sc_1.bmp
Les sous-dossiers train/validation d'une version Kaggle sont fusionnés puis redécoupés.

Les images sont recopiées au format PNG en niveaux de gris : le décodeur BMP de TensorFlow
refuse le mode « 1 canal », or NEU est distribué en BMP.
"""
import argparse
import random
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

# permet de lancer le script directement (python src/prepare_dataset.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    CLASS_NAMES,
    DATA_RAW_DIR,
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    SEED,
)


def _key(text: str) -> str:
    """Normalise un nom : minuscules, uniquement lettres et chiffres."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# alias -> nom de classe canonique
_ALIASES = {}
for _name, _short in zip(CLASS_NAMES, ["cr", "in", "pa", "ps", "rs", "sc"]):
    _ALIASES[_key(_name)] = _name
    _ALIASES[_short] = _name


def guess_class(path: Path):
    """Déduit la classe d'une image depuis son dossier parent, sinon son nom de fichier."""
    parent = _key(path.parent.name)
    if parent in _ALIASES and len(parent) > 2:  # dossier explicite (ex. 'crazing')
        return _ALIASES[parent]
    stem = re.sub(r"[_\-\s]*\d+$", "", path.stem)  # 'Cr_12' -> 'Cr', 'rolled-in_scale_3' -> 'rolled-in_scale'
    return _ALIASES.get(_key(stem))


def collect_images(raw_dir: Path) -> dict:
    by_class = {c: [] for c in CLASS_NAMES}
    unknown = 0
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTENSIONS or not p.is_file():
            continue
        cls = guess_class(p)
        if cls is None:
            unknown += 1
            continue
        by_class[cls].append(p)
    if unknown:
        print(f"[!] {unknown} image(s) ignorée(s) : classe non reconnue.")
    return by_class


def _print_counts(counts: dict) -> None:
    print(f"{'classe':<18}{'train':>7}{'val':>6}{'test':>6}")
    for cls, c in counts.items():
        print(f"{cls:<18}{c['train']:>7}{c['validation']:>6}{c['test']:>6}")


def count_existing(out_dir: Path) -> dict:
    counts = {}
    for cls in CLASS_NAMES:
        counts[cls] = {s: len(list((out_dir / s / cls).glob("*"))) if (out_dir / s / cls).exists() else 0
                       for s in ("train", "validation", "test")}
    return counts


def prepare(raw_dir=DATA_RAW_DIR, out_dir=DATASET_DIR, train=0.70, val=0.15,
            seed=SEED, force=False) -> dict:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    existing = [p for p in (out_dir / s for s in ("train", "validation", "test")) if p.exists()]
    if existing and not force:
        counts = count_existing(out_dir)
        print(f"{out_dir} existe déjà : découpage conservé (utilisez force=True / --force pour le refaire).")
        _print_counts(counts)
        return counts

    if not raw_dir.exists():
        raise SystemExit(f"Dossier introuvable : {raw_dir}\n"
                         "Téléchargez NEU Surface Defect Database et décompressez-le dedans.")

    by_class = collect_images(raw_dir)
    total = sum(len(v) for v in by_class.values())
    if total == 0:
        raise SystemExit(f"Aucune image reconnue dans {raw_dir}.")

    for p in existing:
        shutil.rmtree(p)

    counts = {}
    for cls, files in by_class.items():
        files = sorted(files)
        random.Random(seed).shuffle(files)
        n = len(files)
        n_train, n_val = int(n * train), int(n * val)
        parts = {
            "train": files[:n_train],
            "validation": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for subset, items in parts.items():
            dest = out_dir / subset / cls
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                # préfixe numérique : évite les collisions de noms entre sous-dossiers
                with Image.open(src) as im:
                    im.convert("L").save(dest / f"{i:04d}_{src.stem}.png")
        counts[cls] = {k: len(v) for k, v in parts.items()}

    print(f"\n{total} images réparties (graine = {seed}) :")
    _print_counts(counts)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw", default=str(DATA_RAW_DIR), help="dossier du dataset brut")
    ap.add_argument("--out", default=str(DATASET_DIR), help="dossier de sortie")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="écrase un découpage existant")
    a = ap.parse_args()
    prepare(a.raw, a.out, seed=a.seed, force=a.force)
