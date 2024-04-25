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

# Extraction des valeurs numériques pour la transformation logarithmique
features = paintest_data.drop(columns=['user', 'time', 'is_anomaly'])
numeric_features = features.select_dtypes(include=[np.number])

# Application de la transformation logarithmique
log_transformed_features = np.log1p(numeric_features)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(log_transformed_features)

# Analyse des comportements par jour pour les utilisateurs Paintest
daily_behavior_paintest = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
daily_behavior_paintest['date'] = paintest_data['date']

# Visualisation des tendances temporelles pour les utilisateurs Paintest
plt.figure(figsize=(12, 6))
for date, group in daily_behavior_paintest.groupby('date'):
    plt.scatter(group['PC1'], group['PC2'], label=date)

plt.title('Tendances Temporelles des Comportements Utilisateurs Paintest (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Date')
plt.show()
