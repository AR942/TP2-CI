# CI-with-github

run the code:

    pip3 install -r requirements.txt
    python app.py

the app runs on http://127.0.0.1:5000

test the app:

python -m unittest 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Charger les données (simulées pour l'exemple)
data = pd.read_csv("donnees.csv")

# Sélection des utilisateurs Paintest
paintest_users = ['user1', 'user2', 'user3']  # Remplacer par les utilisateurs Paintest connus

# Filtrer les données pour inclure uniquement les utilisateurs Paintest
paintest_data = data[data['user'].isin(paintest_users)]

# Convertir la colonne 'time' en format datetime
paintest_data['time'] = pd.to_datetime(paintest_data['time'])

# Prétraitement des données avec une transformation logarithmique
features = paintest_data.drop(columns=['user', 'is_anomaly'])
numeric_features = features.select_dtypes(include=[np.number])

# Application de la transformation logarithmique
log_transformed_features = np.log1p(numeric_features)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=1)  # Utilisation d'une seule composante principale
pca_data = pca.fit_transform(log_transformed_features)

# Création du DataFrame daily_behavior_paintest avec la colonne de dates correctement remplie
daily_behavior_paintest = pd.DataFrame(pca_data, columns=['PCA'])
daily_behavior_paintest['hour'] = paintest_data['time'].dt.hour
daily_behavior_paintest['user'] = paintest_data['user'].values

# Visualisation des tendances temporelles pour les utilisateurs Paintest avec les heures sur l'axe des abscisses
plt.figure(figsize=(12, 6))
for user, group in daily_behavior_paintest.groupby('user'):
    plt.plot(group['hour'], group['PCA'], label=f'{user} PCA')

plt.title('Tendances Temporelles des Comportements Utilisateurs Paintest (PCA)')
plt.xlabel('Heure')
plt.ylabel('Valeur PCA')
plt.legend()
plt.show()
