# Prédiction du Diabète via Machine Learning

Ce projet utilise des algorithmes d'apprentissage automatique pour prédire le risque de diabète chez des patients en se basant sur des données médicales (mesures diagnostiques).

## Présentation du Projet
L'objectif est de comparer plusieurs modèles de classification pour identifier le plus performant. Le projet couvre l'intégralité du pipeline de Data Science :
- Nettoyage des données et gestion des valeurs manquantes.
- Mise à l'échelle des variables (Standardization).
- Entraînement et optimisation des hyperparamètres via `GridSearchCV`.
- Évaluation des performances (Précision, Recall, F1-Score).

## Jeu de Données
Les données proviennent du fichier `diabetes.csv` et incluent les indicateurs suivants :
- **Pregnancies** : Nombre de grossesses.
- **Glucose** : Taux de glucose dans le sang.
- **BloodPressure** : Pression artérielle.
- **SkinThickness** : Épaisseur du pli cutané.
- **Insulin** : Taux d'insuline.
- **BMI** : Indice de masse corporelle.
- **DiabetesPedigreeFunction** : Antécédents familiaux.
- **Age** : Âge du patient.
- **Outcome** : Variable cible (0 = non diabétique, 1 = diabétique).

## Modèles Implémentés
Trois algorithmes ont été développés et comparés :
1. **Régression Logistique** : Modèle de référence optimisé avec `liblinear`.
2. **SVM (Support Vector Machine)** : Recherche du meilleur noyau (linéaire, rbf, poly).
3. **Naive Bayes (Gaussian)** : Modèle probabiliste pour la classification.

## Author

Cyrian Luca  
Engineering student in Artificial Intelligence at ESME Sudria
