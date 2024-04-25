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
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_csv("donnees.csv")

# Sélection des utilisateurs Paintest
paintest_users = ['user1', 'user2', 'user3']  # Remplacer par les utilisateurs Paintest connus

# Sélection aléatoire d'un échantillon de utilisateurs non Paintest
non_paintest_users = data[~data['user'].isin(paintest_users)]['user'].unique()
non_paintest_users_sample = np.random.choice(non_paintest_users, size=1000, replace=False)

# Sélection des données pour les utilisateurs Paintest et un échantillon réduit de utilisateurs non Paintest
selected_users = np.concatenate((paintest_users, non_paintest_users_sample))
selected_data = data[data['user'].isin(selected_users)]

# Prétraitement des données
features = selected_data.drop(columns=['user', 'time', 'is_anomaly'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)

# Visualisation des données dans l'espace des composantes principales
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=selected_data['user'].apply(lambda x: 'Paintest' if x in paintest_users else 'Non Paintest'), cmap='coolwarm', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA - Utilisateurs Paintest vs Non Paintest (échantillon réduit)')
plt.colorbar(label='Utilisateur')
plt.show()
