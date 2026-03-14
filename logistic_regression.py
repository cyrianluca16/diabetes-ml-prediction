import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('diabetes.csv', sep=';')

df.fillna(df.median(), inplace=True) #remplace les valeurs manquantes par la médiane

X = df.drop('Outcome', axis=1) #toutes les features
y = df['Outcome'] #la colonne 'Outcome' est la cible

#on utilise 80% des données pour le train set et 20% pour le test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#on normalise les données pour que toutes les features aient des valeurs similaires 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#régression logistique
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

#on évalue le modèle
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

#matrice de confusion
print("\n=== Matrice de Confusion ===")
print(confusion_matrix(y_test, y_pred))

#on donne la précision du modèle en pourcentage
accuracy = logreg.score(X_test_scaled, y_test)
print(f"\nPrécision du modèle : {accuracy * 100:.2f}%")

#on définit la grille des hyperparamètres à tester
param_grid = {'C': [0.01, 0.1, 1, 10, 100],'penalty': ['l1', 'l2'],'solver': ['liblinear']}

grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

print("\n=== Meilleurs hyperparamètres ===")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\n=== Classification Report après optimisation ===")
print(classification_report(y_test, y_pred))

print("\n=== Matrice de confusion après optimisation ===")
print(confusion_matrix(y_test, y_pred))

print(f"\nExactitude : {accuracy_score(y_test, y_pred) * 100:.2f}%")

import joblib

joblib.dump(best_model, "régression_logistique_diabète.pkl")
joblib.dump(scaler, "scaler.pkl")

model = joblib.load("régression_logistique_diabète.pkl")
scaler = joblib.load("scaler.pkl")

features_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

print("Entrez les valeurs du patient :")
input_values = []
for feature in features_names:
    val = float(input(f"{feature} : "))
    input_values.append(val)

features_df = pd.DataFrame([input_values], columns=features_names)
features_scaled = scaler.transform(features_df)

prediction = model.predict(features_scaled)
proba = model.predict_proba(features_scaled)

if prediction[0] == 1:
    print(f"\nLe patient est probablement diabétique (probabilité = {proba[0][1]*100:.1f}%).")
else:
    print(f"\nLe patient n’est probablement pas diabétique (probabilité = {proba[0][0]*100:.1f}%).")
