import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("diabetes.csv", sep=';')

df.fillna(df.median(), inplace=True) #remplace les valeurs manquantes par la médiane

X = df.drop("Outcome", axis=1) #toutes les features
y = df["Outcome"] #la colonne 'Outcome' est la cible

#on utilise 80% des données pour le train set et 20% pour le test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#on normalise les données pour que toutes les features aient des valeurs similaires 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#modèle SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

#on évalue le modèle
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

#matrice de confusion
print("\n=== Matrice de Confusion ===")
print(confusion_matrix(y_test, y_pred))

#on donne la précision du modèle en pourcentage
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision du modèle : {accuracy * 100:.2f}%")

#on définit la grille des hyperparamètres à tester
param_grid = {'C': [0.01, 0.1, 1, 10],'kernel': ['linear', 'rbf', 'poly'],'gamma': ['scale', 'auto']}

grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='f1', verbose=1)
grid.fit(X_train_scaled, y_train)

print("\n=== Meilleurs hyperparamètres ===")
print(grid.best_params_)

best_svm = grid.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

print("\n=== Rapport de classification après optimisation ===")
print(classification_report(y_test, y_pred))

print("\n=== Matrice de confusion après optimisation ===")
print(confusion_matrix(y_test, y_pred))

print(f"\nExactitude : {accuracy_score(y_test, y_pred) * 100:.2f}%")