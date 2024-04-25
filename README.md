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

# Charger les données (simulées pour l'exemple)
data = pd.read_csv("donnees.csv")

# Sélection des utilisateurs Paintest
paintest_users = ['user1', 'user2', 'user3']  # Remplacer par les utilisateurs Paintest connus

# Filtrer les données pour inclure uniquement les utilisateurs Paintest
paintest_data = data[data['user'].isin(paintest_users)]

# Convertir la colonne 'time' en format datetime si nécessaire
paintest_data['time'] = pd.to_datetime(paintest_data['time'])

# Tri des données par ordre chronologique
paintest_data.sort_values(by='time', inplace=True)

# Segmentation des données par jour
paintest_data['date'] = paintest_data['time'].dt.date

# Analyse des comportements par jour pour les utilisateurs Paintest
daily_behavior_paintest = paintest_data.groupby(['user', 'date']).size().reset_index(name='actions_count')

# Visualisation des tendances temporelles pour les utilisateurs Paintest
plt.figure(figsize=(12, 6))
for user in daily_behavior_paintest['user'].unique():
    user_data = daily_behavior_paintest[daily_behavior_paintest['user'] == user]
    plt.plot(user_data['date'], user_data['actions_count'], label=user)

plt.title('Tendances Temporelles des Comportements Utilisateurs Paintest')
plt.xlabel('Date')
plt.ylabel('Nombre d\'actions')
plt.legend()
plt.show()
