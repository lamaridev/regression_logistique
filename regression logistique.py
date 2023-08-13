import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Charger les données avec le bon séparateur
data = pd.read_csv("chemin vers dataset", sep=';')

# Diviser les données en variables indépendantes (X) et variable cible (Y)
X = data[['age', 'salaire']]
Y = data['purchased']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Créer un Standard Scaler
scaler = StandardScaler()

# Normaliser les données d'entraînement et de test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle sur les données d'entraînement normalisées
model.fit(X_train_scaled, Y_train)

# Prédire les valeurs sur les données de test normalisées
Y_pred = model.predict(X_test_scaled)

# Calculer la matrice de confusion
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Matrice de confusion :\n", conf_matrix)
