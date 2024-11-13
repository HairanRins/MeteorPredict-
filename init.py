import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Génération de données synthétiques
np.random.seed(0)
jours = 365 
temperature = np.random.uniform(5,35, jours)  # Température en degrés Celsius
humidite = np.random.uniform(20,80, jours)  # Humidité en pourcentage
vitesse_vent = np.random.uniform(0, 15, jours)  # Vitesse du vent en m/s

# Pollution fictive : associée à une combinaison de température, humidité, et vent
pollution = 50 + 0.5 * temperature - 0.2 * humidite - 0.3 * vitesse_vent + np.random.normal(0, 5, jours)

# Affichage des premières lignes
print(temperature[:5], humidite[:5], vitesse_vent[:5], pollution[:5])

# Préparation des données d'entrée
X = np.column_stack((temperature, humidite, vitesse_vent))
y = pollution

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Prédiction pour un jour spécifique (ex. : 20°C, 50% humidité, 5 m/s de vent)
nouveau_jour = np.array([[20, 50, 5]])
pollution_predite = model.predict(nouveau_jour)

print(f"Pollution estimée pour un jour à 20°C, 50% humidité et 5 m/s de vent : {pollution_predite[0]:.2f} unités")

# Tracé des données
jours = np.arange(1, 366)
plt.plot(jours, pollution, color='blue', label='Pollution réelle')

# Prédiction de la pollution pour chaque jour
pollution_predite = model.predict(X)

# Tracé des prédictions
plt.plot(jours, pollution_predite, color='red', linestyle='--', label='Pollution prédite')
plt.xlabel('Jour')
plt.ylabel('Niveau de pollution')
plt.legend()
plt.title('Niveaux de pollution au cours de l\'année')
plt.show()

# Calcul du MSE et du R²
mse = mean_squared_error(y, pollution_predite)
r2 = r2_score(y, pollution_predite)

print(f'Erreur quadratique moyenne (MSE) : {mse:.2f}')
print(f'Coefficient de détermination (R²) : {r2:.2f}') 

