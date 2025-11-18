# GraphCast & GenCast - Google DeepMind

## Vue d'ensemble du projet

Ce dépôt contient l'implémentation de deux modèles de prévision météorologique de pointe développés par Google DeepMind :

- **GraphCast** : Modèle déterministe de prévision météorologique mondiale à moyen terme, haute résolution, utilisant des réseaux de neurones graphiques
- **GenCast** : Modèle de prévision d'ensemble basé sur la diffusion pour des prédictions météorologiques probabilistes à moyen terme

Ces modèles sont décrits dans les articles de recherche :
- [GraphCast (Science, 2023)](https://www.science.org/doi/10.1126/science.adi2336)
- [GenCast (arXiv, 2023)](https://arxiv.org/abs/2312.15796)

## Ressources disponibles

### Modèles pré-entraînés
- Poids des modèles pré-entraînés
- Statistiques de normalisation
- Exemples de données d'entrée

Disponibles sur [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast)

### Données d'entraînement

**ERA5** - Ensemble de données de réanalyse de l'ECMWF (1979-2018)
- Disponible depuis [ECMWF](https://www.ecmwf.int/)
- Meilleur accès via format Zarr : [Weatherbench2's ERA5 data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#era5)

**HRES-fc0** - Données pour l'ajustement opérationnel
- Disponible via : [Weatherbench2's HRES 0th frame data](https://weatherbench2.readthedocs.io/en/latest/data-guide.html#ifs-hres-t-0-analysis)

⚠️ Ces ensembles de données peuvent être régis par des conditions générales ou des dispositions de licence distinctes.

## Architecture et technologies

Le projet est construit sur **JAX/Haiku** avec :
- Réseaux de neurones graphiques (GNN)
- Transformers pour le traitement des données météorologiques
- Maillages icosaédriques pour la représentation sphérique

### Technologies clés

- **JAX** : Framework de calcul différentiable haute performance
- **Haiku (dm-haiku)** : Bibliothèque de réseaux de neurones
- **Jraph** : Bibliothèque de réseaux de neurones graphiques
- **XArray** : Manipulation de données multidimensionnelles étiquetées
- **Chex** : Utilitaires de test pour JAX
- **Dask** : Calcul parallèle
- **Trimesh** : Opérations sur les maillages
- **Dinosaur** : Noyau dynamique

## Structure du projet

```
graphcast/
├── graphcast/                      # Code source principal
│   ├── graphcast.py                # Architecture du modèle GraphCast
│   ├── gencast.py                  # Architecture du modèle GenCast
│   ├── autoregressive.py           # Wrapper auto-régressif pour l'entraînement
│   ├── rollout.py                  # Déroulement à l'inférence
│   ├── predictor_base.py           # Interface des prédicteurs
│   ├── normalization.py            # Normalisation des données
│   ├── losses.py                   # Fonctions de perte avec pondération par latitude
│   ├── checkpoint.py               # Sérialisation/désérialisation
│   ├── data_utils.py               # Utilitaires de prétraitement
│   │
│   ├── deep_typed_graph_net.py     # GNN profond
│   ├── typed_graph.py              # Définition des TypedGraph
│   ├── typed_graph_net.py          # Blocs de construction GNN
│   ├── sparse_transformer.py       # Transformer sparse pour le maillage
│   ├── sparse_transformer_utils.py # Utilitaires pour transformer sparse
│   ├── transformer.py              # Wrapper du transformer de maillage
│   │
│   ├── grid_mesh_connectivity.py   # Conversion grille ↔ maillage
│   ├── icosahedral_mesh.py         # Définition du multi-maillage icosaédrique
│   ├── model_utils.py              # Production de features vectorielles
│   ├── mlp.py                      # Construction de MLPs avec couches de conditionnement
│   │
│   ├── denoiser.py                 # Débruiteur GenCast
│   ├── denoisers_base.py           # Interface du débruiteur
│   ├── dpm_solver_plus_plus_2s.py  # Échantillonneur DPM-Solver++ 2S
│   ├── samplers_base.py            # Interface de l'échantillonneur
│   ├── samplers_utils.py           # Utilitaires d'échantillonnage
│   ├── nan_cleaning.py             # Gestion des NaN (température de surface)
│   │
│   ├── casting.py                  # Wrapper BFloat16 pour GraphCast
│   ├── solar_radiation.py          # Rayonnement solaire TOA
│   │
│   ├── xarray_jax.py               # Compatibilité JAX ↔ XArray
│   └── xarray_tree.py              # tree.map_structure pour XArray
│
├── docs/                           # Documentation
│   ├── cloud_vm_setup.md           # Configuration VM TPU Google Cloud
│   ├── GenCast_0p25deg_accelerator_scorecard.png
│   └── GenCast_1p0deg_Mini_ENS_scorecard.png
│
├── graphcast_demo.ipynb            # Démo GraphCast (Colab)
├── gencast_mini_demo.ipynb         # Démo GenCast Mini (Colab gratuit)
├── gencast_demo_cloud_vm.ipynb     # Démo GenCast complet (TPU VM)
├── setup.py                        # Configuration du package
└── README.md                       # Documentation principale
```

## Fichiers communs aux deux modèles

### Traitement des graphes
- **`typed_graph.py`** : Définition des `TypedGraph`
- **`typed_graph_net.py`** : Blocs de construction GNN simples pour TypedGraph
- **`deep_typed_graph_net.py`** : GNN profond opérant sur TypedGraph avec vecteurs de features plats

### Gestion du maillage et des grilles
- **`icosahedral_mesh.py`** : Définition d'un multi-maillage icosaédrique
- **`grid_mesh_connectivity.py`** : Conversion entre grilles régulières sur sphère et maillages triangulaires
- **`model_utils.py`** : Production de features vectorielles à partir de données grille et manipulation inverse

### Pipeline de données
- **`data_utils.py`** : Utilitaires de prétraitement des données
- **`normalization.py`** : Normalisation des entrées selon valeurs historiques et des cibles selon différences temporelles
- **`xarray_jax.py`** : Wrapper pour compatibilité JAX avec XArray
- **`xarray_tree.py`** : Implémentation de tree.map_structure pour XArray

### Entraînement et inférence
- **`autoregressive.py`** : Wrapper pour exécuter (et entraîner) les prédictions en une étape en produisant une séquence de prédictions de manière auto-régressive, de façon différentiable en JAX
- **`rollout.py`** : Similaire à autoregressive.py mais utilisé uniquement à l'inférence avec une boucle Python pour produire des trajectoires plus longues mais non différentiables
- **`predictor_base.py`** : Définit l'interface du prédicteur implémentée par tous les modèles et wrappers
- **`losses.py`** : Calculs de perte avec pondération par latitude
- **`checkpoint.py`** : Utilitaires de sérialisation et désérialisation d'arbres

### Utilitaires
- **`mlp.py`** : Construction de MLPs avec couches de conditionnement de normalisation

---

## GenCast : Prévision d'ensemble basée sur la diffusion

GenCast est un modèle de prévision d'ensemble utilisant la diffusion pour générer des prédictions probabilistes de la météo à moyen terme.

### Modèles pré-entraînés disponibles

1. **GenCast 0.25deg <2019**
   - Résolution : 0,25° (haute résolution)
   - 13 niveaux de pression
   - Maillage icosaédrique raffiné 6 fois
   - Entraîné sur ERA5 (1979-2018)
   - Peut être évalué causalement sur 2019 et années ultérieures
   - Modèle décrit dans l'article GenCast

2. **GenCast 0.25deg Operational <2019**
   - Résolution : 0,25°
   - 13 niveaux de pression
   - Maillage icosaédrique raffiné 6 fois
   - Entraîné sur ERA5 (1979-2018)
   - Ajusté sur HRES-fc0 (2016-2021)
   - Peut être évalué causalement sur 2022 et années ultérieures
   - Utilisable en contexte opérationnel (initialisé depuis HRES-fc0)

3. **GenCast 1.0deg <2019**
   - Résolution : 1° (résolution moyenne)
   - 13 niveaux de pression
   - Maillage icosaédrique raffiné 5 fois
   - Entraîné sur ERA5 (1979-2018)
   - Peut être évalué causalement sur 2019 et années ultérieures
   - Empreinte mémoire réduite comparé aux modèles 0,25°

4. **GenCast 1.0deg Mini <2019**
   - Résolution : 1°
   - 13 niveaux de pression
   - Maillage icosaédrique raffiné 4 fois
   - Entraîné sur ERA5 (1979-2018)
   - Peut être évalué causalement sur 2019 et années ultérieures
   - **Plus petite empreinte mémoire** - permet démonstrations à bas coût
   - Exécutable dans un notebook Colab gratuit
   - ⚠️ Performances raisonnables mais non représentatives des modèles GenCast complets (1-3)
   - Scorecard de comparaison avec ENS disponible : [docs/GenCast_1p0deg_Mini_ENS_scorecard.png](https://github.com/google-deepmind/graphcast/blob/main/docs/GenCast_1p0deg_Mini_ENS_scorecard.png)
   - Note : GenCast Mini utilise 8 membres d'ensemble (vs 50 pour ENS), d'où l'utilisation du CRPS équitable (non biaisé) pour comparaison

### Démarrage avec GenCast

**Meilleur point de départ** : Ouvrir `gencast_mini_demo.ipynb` dans [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_mini_demo.ipynb)

Le notebook démontre :
- Chargement des données
- Génération de poids aléatoires ou chargement d'un snapshot GenCast 1.0deg Mini
- Génération de prédictions
- Calcul de la perte et des gradients

**Données et poids** : Disponibles dans le sous-répertoire `gencast/` du Google Cloud Bucket

### Exécution de GenCast sur Google Cloud

Pour exécuter les modèles GenCast complets (1-3), voir :
- [docs/cloud_vm_setup.md](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md) : Instructions détaillées pour lancer une VM TPU Google Cloud
- `gencast_demo_cloud_vm.ipynb` via [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_demo_cloud_vm.ipynb)

### Fichiers spécifiques à GenCast

- **`gencast.py`** : Combine l'architecture GenCast (enveloppée comme débruiteur) avec un échantillonneur pour générer des prédictions
- **`denoiser.py`** : Débruiteur GenCast pour prédictions en une étape
- **`denoisers_base.py`** : Définit l'interface du débruiteur
- **`dpm_solver_plus_plus_2s.py`** : Échantillonneur utilisant DPM-Solver++ 2S [1]
- **`samplers_base.py`** : Définit l'interface de l'échantillonneur
- **`samplers_utils.py`** : Méthodes utilitaires pour l'échantillonneur
- **`sparse_transformer.py`** : Transformer sparse à usage général opérant sur TypedGraph (utilisé pour le GNN de maillage)
- **`sparse_transformer_utils.py`** : Méthodes utilitaires pour le transformer sparse
- **`transformer.py`** : Enveloppe le transformer de maillage, permutant les deux premiers axes des nœuds
- **`nan_cleaning.py`** : Enveloppe un prédicteur pour gérer les données nettoyées des NaN (température de surface de la mer)

**Référence** : [1] DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models, https://arxiv.org/abs/2211.01095

---

## GraphCast : Prévision météorologique mondiale compétente à moyen terme

GraphCast est un modèle déterministe utilisant l'apprentissage profond pour la prévision météorologique mondiale.

### Modèles pré-entraînés disponibles

1. **GraphCast** (modèle haute résolution de l'article)
   - Résolution : 0,25° (haute résolution)
   - 37 niveaux de pression
   - Entraîné sur ERA5 (1979-2017)

2. **GraphCast_small** (version basse résolution)
   - Résolution : 1°
   - 13 niveaux de pression
   - Maillage plus petit
   - Entraîné sur ERA5 (1979-2015)
   - Utile pour contraintes mémoire et calcul réduites

3. **GraphCast_operational** (version opérationnelle)
   - Résolution : 0,25°
   - 13 niveaux de pression
   - Pré-entraîné sur ERA5 (1979-2017)
   - Ajusté sur HRES (2016-2021)
   - Peut être initialisé depuis données HRES (ne nécessite pas d'entrées de précipitation)

### Démarrage avec GraphCast

**Meilleur point de départ** : Ouvrir `graphcast_demo.ipynb` dans [Colaboratory](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb)

Le notebook démontre :
- Chargement des données
- Génération de poids aléatoires ou chargement d'un snapshot pré-entraîné
- Génération de prédictions
- Calcul de la perte et des gradients

**Données et poids** : Disponibles dans le sous-répertoire `graphcast/` du Google Cloud Bucket

⚠️ **Avertissement** : Pour rétrocompatibilité, les données GraphCast sont également disponibles au niveau supérieur du bucket. Ces fichiers seront éventuellement supprimés au profit du sous-répertoire `graphcast/`.

### Fichiers spécifiques à GraphCast

- **`graphcast.py`** : Architecture principale du modèle GraphCast pour une étape de prédictions
- **`casting.py`** : Wrapper autour de GraphCast pour fonctionner en précision BFloat16
- **`solar_radiation.py`** : Calcule le rayonnement solaire incident au sommet de l'atmosphère (TOA) compatible avec ERA5. Utilisé comme variable de forçage, doit être calculé pour les délais cibles en contexte opérationnel

---

## Flux de développement

### 1. Lecture du code du modèle
- Commencer par **`graphcast.py`** ou **`gencast.py`** pour l'architecture
- Vérifier **`predictor_base.py`** pour l'interface commune
- Examiner **`autoregressive.py`** pour comprendre la boucle d'entraînement

### 2. Pipeline de données
- **`data_utils.py`** : Prétraitement
- **`normalization.py`** : Normalisation historique
- **`grid_mesh_connectivity.py`** : Conversion grille → maillage

### 3. Entraînement
- Utiliser **`autoregressive.py`** pour rollouts différentiables
- **`losses.py`** : Pertes pondérées par latitude
- **`checkpoint.py`** : Sérialisation

### 4. Inférence
- Utiliser **`rollout.py`** pour trajectoires longues non différentiables
- Tous les modèles implémentent `predictor_base.Predictor`

---

## Tests

Les fichiers de test suivent le modèle `*_test.py` :

- `checkpoint_test.py`
- `data_utils_test.py`
- `grid_mesh_connectivity_test.py`
- `icosahedral_mesh_test.py`
- `solar_radiation_test.py`
- `xarray_jax_test.py`
- `xarray_tree_test.py`

---

## Dépendances

Bibliothèques principales :

- [Chex](https://github.com/deepmind/chex) - Utilitaires de test JAX
- [Dask](https://github.com/dask/dask) - Calcul parallèle
- [Dinosaur](https://github.com/google-research/dinosaur) - Noyau dynamique
- [Haiku](https://github.com/deepmind/dm-haiku) - Réseaux de neurones
- [JAX](https://github.com/google/jax) - Calcul différentiable
- [JAXline](https://github.com/deepmind/jaxline) - Framework d'entraînement
- [Jraph](https://github.com/deepmind/jraph) - Réseaux de neurones graphiques
- [Numpy](https://numpy.org/) - Calcul numérique
- [Pandas](https://pandas.pydata.org/) - Analyse de données
- [Python](https://www.python.org/) - Langage de programmation
- [SciPy](https://scipy.org/) - Calcul scientifique
- [Tree](https://github.com/deepmind/tree) - Structures arborescentes
- [Trimesh](https://github.com/mikedh/trimesh) - Opérations sur maillages
- [XArray](https://github.com/pydata/xarray) - Tableaux multidimensionnels étiquetés
- [XArray-TensorStore](https://github.com/google/xarray-tensorstore) - Backend de stockage

Voir `setup.py` pour la liste complète des dépendances.

---

## Licences et avertissements

### Licences

**Code (notebooks Colab et code associé)**
- Licence : Apache License, Version 2.0
- Lien : https://www.apache.org/licenses/LICENSE-2.0

**Poids des modèles**
- Licence : Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- Lien : https://creativecommons.org/licenses/by-nc-sa/4.0/
- ⚠️ Usage non commercial uniquement

### Avertissements importants

- ❌ **Pas un produit officiellement supporté par Google**
- 🧪 **Projet de recherche expérimental**
- ⚠️ **Fourni "TEL QUEL"** sans garanties ni conditions d'aucune sorte

### Responsabilités

Vous êtes **seul responsable** de :
- Déterminer si l'utilisation de GenCast/GraphCast est appropriée
- Tous les risques associés à votre utilisation ou distribution
- L'exercice des droits et permissions accordés par les licences

### Utilisation prudente recommandée

GenCast et GraphCast ou toutes sorties générées :
- ❌ Ne sont **pas basés** sur des données publiées par des agences météorologiques gouvernementales
- ❌ N'ont **pas été produits** en collaboration avec ces agences
- ❌ N'ont **pas été approuvés** par ces agences
- ❌ Ne **remplacent en aucun cas** les alertes, avertissements ou avis officiels

**Conseil** : Faire preuve de discernement avant de se fier à, publier, télécharger ou utiliser GenCast, GraphCast ou toute sortie générée.

### Conformité des données

Les données ERA5 et HRES sont soumises à des conditions générales distinctes :

**Données ERA5**
- Service Copernicus sur le changement climatique (modifié, 2023)
- Ni la Commission européenne ni l'ECMWF ne sont responsables de l'utilisation des informations ou données Copernicus

**Données HRES de l'ECMWF**
- Copyright : "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)"
- Source : www.ecmwf.int
- Licence : Creative Commons Attribution 4.0 International (CC BY 4.0)
- Lien : https://creativecommons.org/licenses/by/4.0/
- Avertissement : L'ECMWF n'accepte aucune responsabilité pour les erreurs, omissions, disponibilité ou dommages découlant de l'utilisation des données

⚠️ Vérifiez la conformité avec les politiques de données de l'ECMWF avant utilisation.

---

## Citations

Lors de l'utilisation de ce code, veuillez citer les articles suivants :

### GraphCast

**Article Science** : [Learning skillful medium-range global weather forecasting](https://www.science.org/doi/10.1126/science.adi2336)

```latex
@article{lam2023learning,
  title={Learning skillful medium-range global weather forecasting},
  author={Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and others},
  journal={Science},
  volume={382},
  number={6677},
  pages={1416--1421},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

### GenCast

**Article arXiv** : [GenCast: Diffusion-based ensemble forecasting for medium-range weather](https://arxiv.org/abs/2312.15796)

```latex
@article{price2023gencast,
  title={GenCast: Diffusion-based ensemble forecasting for medium-range weather},
  author={Price, Ilan and Sanchez-Gonzalez, Alvaro and Alet, Ferran and Andersson, Tom R and El-Kadi, Andrew and Masters, Dominic and Ewalds, Timo and Stott, Jacklynn and Mohamed, Shakir and Battaglia, Peter and Lam, Remi and Willson, Matthew},
  journal={arXiv preprint arXiv:2312.15796},
  year={2023}
}
```

---

## Remerciements

GenCast et GraphCast communiquent avec et/ou référencent les bibliothèques et packages séparés mentionnés ci-dessus.

Les notebooks Colab incluent quelques exemples de données ERA5 et HRES de l'ECMWF pouvant être utilisées comme entrées pour les modèles.

**Données et produits** : European Centre for Medium-range Weather Forecasts (ECMWF), modifiés par Google

L'utilisation des matériaux tiers mentionnés ci-dessus peut être régie par des conditions générales ou dispositions de licence distinctes. Vérifiez la conformité avec les restrictions ou conditions applicables avant utilisation.

---

## Contact

Pour commentaires et questions : **gencast@google.com**

---

## Ressources supplémentaires

### Documentation
- [Configuration VM Cloud](docs/cloud_vm_setup.md)
- [Scorecard GenCast 0.25deg Accelerator](docs/GenCast_0p25deg_accelerator_scorecard.png)
- [Scorecard GenCast 1.0deg Mini vs ENS](docs/GenCast_1p0deg_Mini_ENS_scorecard.png)

### Publications
- [Blog DeepMind](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/)
- [Article GraphCast (Science)](https://www.science.org/doi/10.1126/science.adi2336)
- [Article GraphCast (arXiv)](https://arxiv.org/abs/2212.12794)
- [Article GenCast (arXiv)](https://arxiv.org/abs/2312.15796)

### Notebooks Colab
- [Démo GraphCast](https://colab.research.google.com/github/deepmind/graphcast/blob/master/graphcast_demo.ipynb)
- [Démo GenCast Mini](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_mini_demo.ipynb)
- [Démo GenCast Cloud VM](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_demo_cloud_vm.ipynb)

---

**Copyright 2024 DeepMind Technologies Limited**
