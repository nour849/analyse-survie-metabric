#  Analyse de la Durée de Survie et Plateforme Interactive — Dataset METABRIC

---

##  1. Présentation du Projet
Ce projet est une étude complète de statistique inférentielle dédiée à la modélisation de la durée de survie des patientes atteintes du cancer du sein, basée sur la cohorte internationale **METABRIC** (*Molecular Taxonomy of Breast Cancer International Consortium*).

L'objectif est double :
1. **Analyse Inférentielle (Notebook) :** Comparer rigoureusement l'ajustement empirique (Kaplan-Meier) à plusieurs lois paramétriques (Exponentielle, Weibull, Gompertz, Gamma) par maximum de vraisemblance (MLE) et critères d'information (AIC/BIC).
2. **Vulgarisation Interactive (Dashboard) :** Fournir une application web interactive permettant d'explorer graphiquement la cohorte, de filtrer les profils et de simuler dynamiquement les courbes de survie.

---

##  2. Structure du Dépôt GitHub

L'organisation du projet se présente de la manière suivante :
```text
analyse-survie-metabric/
│
├── data/
│   └── Breast Cancer METABRIC.csv       # Dataset téléchargé depuis Kaggle
│
├──  projet_survie_finale.ipynb          # Analyse statistique et calculs MLE/AIC/BIC
├── app.py                               # Code de l'application Dashboard Streamlit
├── requirements.txt                     # Liste des dépendances Python requises
└── README.md                            # Documentation du projet 