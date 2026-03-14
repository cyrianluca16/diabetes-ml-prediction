import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv", sep=';')

df.fillna(df.median(), inplace=True) #remplace les valeurs manquantes par la médiane

X = df.drop('Outcome', axis=1) #toutes les features
y = df['Outcome'] #la colonne 'Outcome' est la cible

#on utilise 80% des données pour le train set et 20% pour le test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#on normalise les données pour que toutes les features aient des valeurs similaires 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#modèle de Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

y_pred = nb.predict(X_test_scaled)

#on évalue le modèle
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

#matrice de confusion
print("\n=== Matrice de Confusion ===")
print(confusion_matrix(y_test, y_pred))

#on donne la précision du modèle en pourcentage
accuracy = nb.score(X_test_scaled, y_test)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#on définit les valeurs de var_smoothing à tester
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

grid_search_nb = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='f1')

grid_search_nb.fit(X_train_scaled, y_train)

print("Meilleurs hyperparamètres Naive Bayes :")
print(grid_search_nb.best_params_)

best_nb_model = grid_search_nb.best_estimator_
y_pred_nb = best_nb_model.predict(X_test_scaled)

print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred_nb))

print("\n=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred_nb))

print(f"\nExactitude : {accuracy_score(y_test, y_pred_nb) * 100:.2f}%")

results = pd.DataFrame(grid_search_nb.cv_results_)
print(results[['param_var_smoothing', 'mean_test_score', 'std_test_score']])
