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
from sklearn.manifold import TSNE

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

# Prétraitement des données avec une transformation logarithmique
features = selected_data.drop(columns=['user', 'time', 'is_anomaly'])
log_transformed_features = np.log1p(features)  # Appliquer une transformation logarithmique en évitant les valeurs nulles

# Réduction de dimensionnalité avec t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_data = tsne.fit_transform(log_transformed_features)

# Assigner une couleur différente aux utilisateurs Paintest et non Paintest
colors = ['blue' if user in paintest_users else 'red' for user in selected_data['user']]

# Visualisation des données dans l'espace t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=colors, alpha=0.5)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE - Utilisateurs Paintest vs Non Paintest (échantillon réduit)')
plt.show()
