# GraphCast & GenCast

## Aperçu du Projet

Ce dépôt contient l'implémentation de deux modèles de prévision météorologique de pointe développés par Google DeepMind :

- **GraphCast** : Un modèle déterministe de prévision météorologique haute résolution utilisant des réseaux de neurones graphiques
- **GenCast** : Un modèle de prévision d'ensemble basé sur la diffusion pour des prédictions météorologiques probabilistes

Les deux modèles fournissent des prévisions météorologiques mondiales à moyen terme en utilisant l'apprentissage automatique entraîné sur les données de réanalyse ERA5.

## Architecture

Le projet est construit sur JAX/Haiku avec des réseaux de neurones graphiques et des transformers pour traiter les données météorologiques sur des maillages icosaédriques.

### Technologies Clés

- **JAX** : Framework de calcul principal
- **Haiku** : Bibliothèque de réseaux de neurones
- **Jraph** : Bibliothèque de réseaux de neurones graphiques
- **XArray** : Manipulation de données multidimensionnelles
- **Chex** : Utilitaires de test pour JAX

## Structure du Projet

```
graphcast/
├── graphcast/          # Code source principal
│   ├── graphcast.py    # Architecture du modèle GraphCast
│   ├── gencast.py      # Architecture du modèle GenCast
│   ├── autoregressive.py        # Wrapper auto-régressif pour l'entraînement
│   ├── rollout.py              # Déroulement au moment de l'inférence
│   ├── normalization.py        # Normalisation des données
│   ├── losses.py               # Fonctions de perte
│   ├── deep_typed_graph_net.py # Implémentation GNN profonde
│   ├── sparse_transformer.py   # Transformer sparse pour le traitement du maillage
│   ├── denoiser.py             # Débruiteur GenCast
│   ├── dpm_solver_plus_plus_2s.py # Échantillonneur DPM-Solver++
│   ├── grid_mesh_connectivity.py  # Conversion grille-maillage
│   ├── icosahedral_mesh.py     # Définition du maillage
│   └── ...
├── docs/               # Documentation et tableaux de bord
├── *.ipynb            # Notebooks de démonstration
├── setup.py           # Configuration du package
└── README.md          # Documentation principale
```

## Composants Principaux

### Modèle GraphCast

**Fichiers principaux :**
- `graphcast.py` - Architecture de prédiction en une étape
- `casting.py` - Wrapper de précision BFloat16
- `solar_radiation.py` - Rayonnement solaire incident au sommet de l'atmosphère (TOA)

**Modèles pré-entraînés disponibles :**
1. GraphCast (0,25°, 37 niveaux) - entraîné sur 1979-2017
2. GraphCast_small (1°, 13 niveaux) - entraîné sur 1979-2015
3. GraphCast_operational (0,25°, 13 niveaux) - ajusté sur HRES

### Modèle GenCast

**Fichiers principaux :**
- `gencast.py` - Architecture du modèle avec débruiteur et échantillonneur
- `denoiser.py` - Débruiteur en une étape
- `denoisers_base.py` - Interface du débruiteur
- `samplers_base.py`, `samplers_utils.py` - Interface et utilitaires d'échantillonnage

**Modèles pré-entraînés disponibles :**
1. GenCast 0,25deg <2019 - Haute résolution ERA5
2. GenCast 0,25deg Operational <2019 - Ajusté sur HRES
3. GenCast 1,0deg <2019 - Empreinte mémoire réduite
4. GenCast 1,0deg Mini <2019 - Modèle le plus petit pour les démonstrations

### Composants Partagés

- **Opérations de Graphe** : `typed_graph.py`, `typed_graph_net.py`, `deep_typed_graph_net.py`
- **Gestion du Maillage** : `icosahedral_mesh.py`, `grid_mesh_connectivity.py`
- **Traitement des Données** : `data_utils.py`, `xarray_jax.py`, `xarray_tree.py`
- **Utilitaires d'Entraînement** : `autoregressive.py`, `normalization.py`, `losses.py`
- **Utilitaires de Modèle** : `model_utils.py`, `mlp.py`, `checkpoint.py`

## Exigences en Matière de Données

### Données d'Entraînement
- **ERA5** : Ensemble de données de réanalyse de l'ECMWF (1979-2018)
  - Disponible via Weatherbench2 au format Zarr
  - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5

### Données d'Ajustement
- **HRES-fc0** : Données d'analyse opérationnelle (2016-2021)
  - Disponible via Weatherbench2
  - https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis

### Ressources Pré-entraînées
- Poids des modèles, statistiques de normalisation et exemples de données disponibles à :
  - Google Cloud Bucket : `gs://dm_graphcast`

## Démarrage

### Notebooks de Démonstration

1. **Démo GraphCast** (`graphcast_demo.ipynb`)
   - Chargement de modèles pré-entraînés
   - Génération de prédictions
   - Calcul de la perte et des gradients
   - Peut s'exécuter dans Colab

2. **Démo GenCast Mini** (`gencast_mini_demo.ipynb`)
   - Exécution du plus petit modèle GenCast
   - Adapté aux notebooks Colab gratuits
   - Bon point de départ pour comprendre le code

3. **Démo GenCast Cloud VM** (`gencast_demo_cloud_vm.ipynb`)
   - Exécution des modèles GenCast pleine taille
   - Nécessite une VM TPU Google Cloud
   - Voir `docs/cloud_vm_setup.md` pour les instructions de configuration

### Flux de Développement

1. **Lecture du Code du Modèle**
   - Commencer par `graphcast.py` ou `gencast.py` pour l'architecture du modèle
   - Vérifier `predictor_base.py` pour l'interface implémentée par tous les modèles
   - Examiner `autoregressive.py` pour comprendre la boucle d'entraînement

2. **Pipeline de Données**
   - `data_utils.py` - Utilitaires de prétraitement
   - `normalization.py` - Normalisation historique
   - `grid_mesh_connectivity.py` - Conversion de grilles en maillage

3. **Entraînement**
   - Utiliser le wrapper `autoregressive.py` pour les déroulements différentiables
   - `losses.py` fournit des calculs de perte pondérés par latitude
   - `checkpoint.py` pour la sérialisation

4. **Inférence**
   - Utiliser `rollout.py` pour des trajectoires plus longues non différentiables
   - Les modèles implémentent l'interface `predictor_base.Predictor`

## Dépendances

Dépendances clés (voir `setup.py` pour la liste complète) :
- `jax` - Calcul principal
- `dm-haiku` - Framework de réseaux de neurones
- `jraph` - Réseaux de neurones graphiques
- `xarray` - Tableaux multidimensionnels
- `numpy`, `scipy` - Calcul numérique
- `trimesh` - Opérations de maillage
- `dask` - Calcul parallèle
- `chex` - Utilitaires de test JAX
- `dinosaur-dycore` - Noyau dynamique

## Tests

Les fichiers de test suivent le modèle `*_test.py` :
- `checkpoint_test.py`
- `data_utils_test.py`
- `grid_mesh_connectivity_test.py`
- `icosahedral_mesh_test.py`
- `solar_radiation_test.py`
- `xarray_jax_test.py`
- `xarray_tree_test.py`

## Notes Importantes

### Licences
- **Code** : Apache License 2.0
- **Poids des Modèles** : CC BY-NC-SA 4.0 (non commercial)

### Conformité des Données
- Les données ERA5 et HRES ont des conditions générales distinctes
- Vérifier la conformité avec les politiques de données de l'ECMWF avant utilisation
- Informations modifiées du service Copernicus sur le changement climatique

### Avertissements
- Pas un produit officiellement supporté par Google
- Projet de recherche expérimental
- Non approuvé par les agences météorologiques
- Ne remplace pas les alertes/avertissements météorologiques officiels

## Citations

Lors de l'utilisation de ce code, citer :

**GraphCast :**
```
Lam et al. (2023). Learning skillful medium-range global weather forecasting.
Science, 382(6677), 1416-1421.
```

**GenCast :**
```
Price et al. (2023). GenCast: Diffusion-based ensemble forecasting for
medium-range weather. arXiv preprint arXiv:2312.15796.
```

## Contact

Pour les commentaires et questions : gencast@google.com

## Ressources Supplémentaires

- Article de blog : https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/
- Article GraphCast : https://www.science.org/doi/10.1126/science.adi2336
- Article GenCast : https://arxiv.org/abs/2312.15796
- Configuration Cloud VM : `docs/cloud_vm_setup.md`
- Tableaux de bord : répertoire `docs/`
