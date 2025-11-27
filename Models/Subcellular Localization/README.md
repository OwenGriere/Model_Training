# Subcellular Localization

Ce dépôt propose un pipeline complet dédié à l’apprentissage, l’évaluation et l’analyse de modèles de classification de localisation subcellulaire à partir de données séquentielles. Le projet repose sur **Theano** et **Lasagne**, tout en offrant plusieurs architectures de réseaux neuronaux, un mode de recherche d’hyperparamètres, ainsi que des notebooks d’exploration des données.

---

## Présentation générale

L’objectif est de prédire la localisation subcellulaire d’une protéine à partir d’une représentation séquentielle (embeddings par résidus).  
Le pipeline inclut :

- plusieurs architectures : **FFN**, **CNN**, **CNN‑LSTM**, **CNN‑LSTM + Attention** ;
- un système d’entraînement configurable via YAML ;
- un mode **multimodel/grid‑search** automatisé ;
- des outils de visualisation (loss, matrices de confusion) ;
- un historique complet des runs dans un fichier unique `.parquet` ;
- des notebooks pour l’analyse des données et des résultats.

---

## Structure de l'outil

```
Subcellular Localization/
├── train.py                    # Entraînement (simple ou grid-search)
├── test.py                     # Évaluation d’un modèle sauvegardé
├── config/
│   ├── simple_model.yaml       # Configuration pour un seul modèle
│   └── multimodels.yaml        # Configuration pour un grid-search
├── data/                       # Données sous forme .npz
├── models/                     # Modèles sauvegardés (.npz)
├── params/
│   └── params.parquet          # Historique de tous les entraînements
├── Figures/                    # Courbes et matrices de confusion
├── mypackage/                  # Modules internes (modèles, utils, plots)
├── analyse_embeddings.ipynb    # Analyse exploratoire des embeddings
├── param_analyse.ipynb         # Analyse UMAP des hyperparamètres
└── env.yml                     # Environnement conda
```

---

## Installation et environnement

Le projet utilise **Python 3.8**. L’environnement conda fourni installe Theano, Lasagne et le reste des dépendances.

### Installation

```bash
conda env create -f env.yml
conda activate theano
```

### Vérification

```bash
python -c "import theano, lasagne; print('Environnement opérationnel')"
```

---

## Format des données

Les jeux de données sont fournis au format `.npz` et doivent contenir :

- `X_*` : tableau `(N, seq_len, n_features)`  
- `y_*` : labels entiers `(N,)`  
- `mask_*` : masque `(N, seq_len)` pour gérer les séquences de longueurs variables

Fichiers attendus dans `data/` :

```
train.npz
reduced_train.npz
reduced_val.npz
test.npz
```

Les masques sont essentiels pour les modèles séquentiels.

---

## Entraînement d’un modèle

L’entraînement standard se fait via `train.py`.

### Commande principale

```bash
python train.py   --config_path ./config/simple_model.yaml   --params_path ./params/params.parquet   --saving   --verbose
```

### Options utiles

| Option | Rôle |
|--------|------|
| `--saving` | sauvegarde du modèle dans `models/` |
| `--no_plot` | désactive la génération des figures |
| `--verbose` | affiche les messages principaux |
| `--all_verbose` | détail des epochs |

### Sorties générées

- modèle `.npz` dans `models/`
- dossier de figures dans `Figures/ID_xxx/`
- nouvelle entrée dans `params/params.parquet`

---

## 6. Recherche d’hyperparamètres (Grid‑Search)

Le fichier `config/multimodels.yaml` définit toutes les combinaisons à explorer.

### Commande

```bash
python train.py   --multimodel   --params_path ./params/params.parquet   --saving
```

### Fonctionnement

- toutes les combinaisons de `multimodels.yaml` sont évaluées ;
- chaque run reçoit un ID unique ;
- un modèle, des figures et des métriques sont générés pour chaque combinaison.

---

## Évaluation d’un modèle sauvegardé

Un modèle `.npz` peut être réévalué sur le jeu de test via `test.py`.

```bash
python test.py   --config_path ./config/simple_model.yaml   --model_path ./models/101_CNN-LSTM.npz   --verbose
```

### Sorties produites

- matrice de confusion
- courbes loss/accuracy
- métriques (accuracy, Gorodkin, IC…)

---

## Analyse des résultats

### Analyse des embeddings

Notebook :

```bash
jupyter notebook analyse_embeddings.ipynb
```

Utilisé pour :

- inspecter la distribution des classes,
- étudier les embeddings,
- visualiser les représentations internes d’un modèle.

### Analyse des hyperparamètres

Notebook :

```bash
jupyter notebook param_analyse.ipynb
```

Outils :

- chargement du fichier parquet,
- filtrage et normalisation des hyperparamètres,
- réduction de dimension UMAP,
- analyse globale des performances.

---

## Architectures disponibles

Définies dans `mypackage/models.py` :

- **FFN** : réseau fully connected
- **CNN** : convolution 1D
- **CNN‑LSTM** : convolution + récurrence
- **CNN‑LSTM‑Attention** : version améliorée avec mécanisme d’attention

Les hyperparamètres clé sont réglés via les fichiers YAML.

---

## Workflow complet recommandé

```bash
# 1. Entraînement d’un modèle
python train.py -c config/simple_model.yaml --saving --verbose

# 2. Évaluation
python test.py --config_path config/simple_model.yaml                --model_path models/101_CNN-LSTM.npz                --verbose

# 3. Grid-search
python train.py --multimodel --saving

# 4. Analyse des résultats avec UMAP
jupyter notebook param_analyse.ipynb
```
