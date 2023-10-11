import unittest
from app import app

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_hello(self):
        response = self.app.get('/api/hello')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'hello': 'world'})

    def test_hello_name(self):
        response = self.app.get('/api/hello/ben')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'hello': 'ben'})

    def test_whoami(self):
        response = self.app.get('/api/whoami')
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json['ip'])

    def test_whoami_name(self):
        response = self.app.get('/api/whoami/ben')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['name'], 'ben')

if __name__ == '__main__':
    unittest.main()


import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Charger les données depuis un fichier CSV (assurez-vous de remplacer 'votre_fichier.csv' par le chemin vers votre fichier de données)
data = pd.read_csv('votre_fichier.csv')

# Assurez-vous que '_time' est de type datetime
data['_time'] = pd.to_datetime(data['_time'])

# Trier les données par date
data.sort_values(by='_time', inplace=True)

# Créez un DataFrame séparé pour les prédictions futures (après le 7 octobre 2023)
future_dates = pd.date_range(start='2023-10-08', end='2023-10-10', freq='D')
future_data = pd.DataFrame({'_time': future_dates})

# Séparez les fonctionnalités (dates) et les étiquettes (nombre de licences)
X = data['_time'].values.reshape(-1, 1)
y = data['total'].values

# Créez des polynômes de degré 2 (vous pouvez ajuster cela en fonction de la complexité de vos données)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Entraînez un modèle de régression linéaire avec les polynômes
model = LinearRegression()
model.fit(X_poly, y)

# Préparez les données de prédiction pour les futures dates
X_future = future_data['_time'].values.reshape(-1, 1)
X_future_poly = poly.transform(X_future)

# Faites des prédictions
predictions = model.predict(X_future_poly)

# Tracez les prédictions
plt.figure(figsize=(12, 6))
plt.scatter(data['_time'], y, label='Données réelles', color='blue')
plt.plot(future_data['_time'], predictions, label='Prédictions', color='red')
plt.xlabel('Date')
plt.ylabel('Nombre de licences')
plt.legend()
plt.show()

# Affichez les prédictions pour les dates futures
for date, prediction in zip(future_data['_time'], predictions):
    print(f"Date: {date}, Prédiction: {prediction}")
