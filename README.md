Title

## Data science ML: Détection du support oblique haussier dominant via RANSAC et du 1er break de support — étude sur l'action TotalEnergies (2021–2026)

### 1. Projet

Mise en oeuvre d'un pipeline quantitative pour détecter les structures de support oblique haussier dominantes et identifier les breaks de support par le bas confirmées.

L'étude porte sur TotalEnergies (TTE.PA) sur la période 2021-2026, avec une séparation temporelle stricte :

Période d'entraînement : 2021-2023

Période de test : 2024-2026

L'objectif: transformer un concept classique d'analyse technique en une méthodologie reproductible et statistiquement robuste.

### 2. Méthodologie

4 étapes dans le pipeline

#### 2.1 Détection des points bas du cours de l'action par fractales

utilisation de la méthode de Bill Williams, version 5 bars

 Point i est fractal si :
        Low[i] < Low[i-1]
        Low[i] < Low[i-2]
        Low[i] < Low[i+1]
        Low[i] < Low[i+2]


on obtient un lot de candidats potentiels pour tracer les supports

#### 2.2 Estimation (tracer) du support oblique haussier dominant via RANSAC

utilisation de la méthode RANSAC avec estimateur linéaire

Pourquoi RANSAC ?

Identifier de manière robuste les supports obliques haussiers au milieu des points fractales dont une partie sont du bruit.

C’est un outil de robustesse géométrique.

Identification de l’alignement structurel dominant

Évite le tracé subjectif des lignes de tendance

Contraintes du modèle :

Pente positive uniquement (support haussier)

Nombre minimal d'inliers ≥ 10

Nombre d'inliers ≥ 15 % des fractales détectées

Robustesse ajustée de la volatilité

Le seuil résiduel est défini comme suit :

seuil_résiduel = 1 × ATR moyenne calculé sur la période d'entraînement

pour faire face à la volatilité.

#### 2.3 Projection du support sur une période de test

La ligne de support détectée (2021-2023) est projetée sur la période 2024-2026 sans réajustement.

Ceci garantit une validation hors échantillon.

#### 2.4 Break du support oblique haussier par le bas, conditions

Pénétration du support > 1 %

Clôture sous le support

Cette condition persiste pendant 3 jours consécutifs.

L'idée c'est de filtrer les faux breaks.

### 3. Outputs

Support oblique haussier dominant (si détecté)

Points bas estimés par la méthode des fractales de Bill Williams

Break du support par le bas confirmé

Visualisation 

### 4. Repository Structure

```text
project/
│
├── data/
├── src/
│   ├── fractal.py
│   ├── atr.py
│   ├── ransac_support.py
│   ├── break_detection.py
│   └── plot.py
│
├── notebook_demo.ipynb
├── requirements.txt
└── README.md
```

### 5. Stack technique

Python
pandas
numpy
scikit-learn
matplotlib

### 6. Principes clés

Cas d'usage RANSAC ML simple

Strict train/test split (pas de fuite d'informations)

Régression ransac plutôt que les moindres carrés ordinaires

Threshold ajusté de la volatilité

Modélisation déterministe

Pas d'optimisation des hyper paramètres

### 7. Limites

Une seule action étudiée

Un seul support dominant mis en évidence

Pas de segmentation et de changements de régimes

Ce mini projet privilégie la clarté méthodologique aux prédictions.

### 8. Future Extensions possibles

Multi-asset généralisation

détection des changements de régime

Score de pénétration du support

Apprentissage supervisé

### 9. Motivation

Ce projet vise à démontrer comment les concepts classiques de l'analyse technique peuvent être traduits en 

Définitions quantitatives,
Méthodes statistiques robustes,
Validation hors échantillon