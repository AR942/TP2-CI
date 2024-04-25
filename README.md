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

# Tri des données par ordre chronologique
paintest_data.sort_values(by='time', inplace=True)

# Segmentation des données par jour
paintest_data['date'] = paintest_data['time'].dt.date

# Prétraitement des données avec une transformation logarithmique
features = paintest_data.drop(columns=['user', 'time', 'is_anomaly'])
numeric_features = features.select_dtypes(include=[np.number])

# Application de la transformation logarithmique
log_transformed_features = np.log1p(numeric_features)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(log_transformed_features)

# Création du DataFrame daily_behavior_paintest avec la colonne de dates correctement remplie
daily_behavior_paintest = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
daily_behavior_paintest['date'] = paintest_data['date'].values  # Utiliser les valeurs de la colonne 'date' sans conversion
daily_behavior_paintest['user'] = paintest_data['user'].values

# Visualisation des tendances temporelles pour les utilisateurs Paintest avec des lignes continues
plt.figure(figsize=(12, 6))
for user, group in daily_behavior_paintest.groupby('user'):
    plt.plot(group['date'], group['PC1'], label=f'{user} PC1')
    plt.plot(group['date'], group['PC2'], label=f'{user} PC2')

plt.title('Tendances Temporelles des Comportements Utilisateurs Paintest (PCA)')
plt.xlabel('Date')
plt.ylabel('Valeur PCA')
plt.legend()
plt.show() 



