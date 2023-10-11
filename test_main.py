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
from workalendar.europe import France
import holidays

# Charger les données depuis un fichier CSV (assurez-vous de remplacer 'votre_fichier.csv' par le chemin vers votre fichier de données)
data = pd.read_csv('votre_fichier.csv')

# Assurez-vous que '_time' est de type datetime
data['_time'] = pd.to_datetime(data['_time'])

# Créer un calendrier pour les jours fériés
fr_cal = France()
fr_holidays = holidays.France(years=range(2022, 2024))

# Trier les données par date
data.sort_values(by='_time', inplace=True)

# Ajustement de la valeur "total" pour les jours précédents
data['total'] = data['total'].shift(1)

# Ajouter des colonnes pour les jours de la semaine (week-end), les jours fériés et les vacances
data['_day_of_week'] = data['_time'].dt.dayofweek  # 0 = Lundi, 6 = Dimanche
data['_is_weekend'] = data['_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 1 pour week-end, 0 sinon
data['_is_holiday'] = data['_time'].apply(lambda x: 1 if x in fr_holidays else 0)
data['_is_vacation'] = data['_time'].apply(lambda x: 1 if fr_cal.is_working_day(x) == False else 0)

# Définir un seuil aberrant
seuil_aberrant = 1000

# Filtrer les valeurs aberrantes
data = data[data['total'] <= seuil_aberrant]

# Créez un DataFrame séparé pour les prédictions futures (après le 7 octobre 2023)
future_dates = pd.date_range(start='2023-10-08', end='2023-10-10', freq='D')
future_data = pd.DataFrame({'_time': future_dates})

# Séparez les fonctionnalités (dates, jours de la semaine, jours fériés, vacances) et les étiquettes (nombre de licences)
X = data[['_time', '_day_of_week', '_is_weekend', '_is_holiday', '_is_vacation']].values
y = data['total'].values

# Créez des polynômes de degré 2 (vous pouvez ajuster cela en fonction de la complexité de vos données)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Entraînez un modèle de régression linéaire avec les polynômes
model = LinearRegression()
model.fit(X_poly, y)

# Préparez les données de prédiction pour les futures dates, en incluant les caractéristiques correspondantes
future_data['_day_of_week'] = future_data['_time'].dt.dayofweek
future_data['_is_weekend'] = future_data['_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
future_data['_is_holiday'] = future_data['_time'].apply(lambda x: 1 if x in fr_holidays else 0)
future_data['_is_vacation'] = future_data['_time'].apply(lambda x: 1 if fr_cal.is_working_day(x) == False else 0)
X_future = future_data[[' _day_of_week', '_is_weekend', '_is_holiday', '_is_vacation']].values
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
