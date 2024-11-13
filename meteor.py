import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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


# Prédiction pour 10h30 le dimanche (ex. : 19°C, 58% humidité, 5.833 m/s de vent)
nouveau_jour = np.array([[19, 58,  5.833]])
pollution_predite_10h30 = model.predict(nouveau_jour)

# Prédiction à 11h : Température = 21°C, Humidité = 51%, Vent = 5.833 m/s (21 km/h)
nouveau_jour_11h = np.array([[21, 51, 5.833]])
pollution_predite_11h = model.predict(nouveau_jour_11h)

# Prédiction à 12h : Température = 22°C, Humidité = 45%, Vent = 5.833 m/s (21 km/h)
nouveau_jour_12h = np.array([[22, 45, 5.833]])
pollution_predite_12h = model.predict(nouveau_jour_12h)

# Prédiction à 13h : Température = 22°C, Humidité = 42%, Vent = 6.389 m/s (23 km/h)
vitesse_vent_13h = 23 / 3.6  # Convertir la vitesse du vent de km/h en m/s
nouveau_jour_13h = np.array([[22, 42, vitesse_vent_13h]])
pollution_predite_13h = model.predict(nouveau_jour_13h)

# Prédiction à 14h : Température = 23°C, Humidité = 38%, Vent = 6.389 m/s (23 km/h)
nouveau_jour_14h = np.array([[23, 38, vitesse_vent_13h]])
pollution_predite_14h = model.predict(nouveau_jour_14h)

print(f"Pollution estimée à 10h30 pour le 13/10/2024 , à 19°C, 58% humidité et 5.833 m/s de vent : {pollution_predite_10h30[0]:.2f} µg/m³ de PM2.5")
print(f"Pollution estimée à 11h pour le 13/10/2024 , à 21°C, 51% humidité et 5.833 m/s de vent : {pollution_predite_11h[0]:.2f} µg/m³ de PM2.5")
print(f"Pollution estimée à 12h pour le 13/10/2024 , à 22°C, 45% humidité et 5.833 m/s de vent : {pollution_predite_12h[0]:.2f} µg/m³ de PM2.5")
print(f"Pollution estimée à 13h pour le 13/10/2024 , à 22°C, 42% humidité et {vitesse_vent_13h:.3f} m/s de vent : {pollution_predite_13h[0]:.2f} µg/m³ de PM2.5")
print(f"Pollution estimée à 14h pour le 13/10/2024 , 23°C, 38% humidité et {vitesse_vent_13h:.3f} m/s de vent : {pollution_predite_14h[0]:.2f} µg/m³ de PM2.5")
# la valeur de pollution prédite qu'on obtient représente une estimation de la concentration de polluants dans l'air



heures = ['10h30', '11h', '12h', '13h', '14h']
pollution_values = [
    pollution_predite_10h30[0],
    pollution_predite_11h[0],
    pollution_predite_12h[0],
    pollution_predite_13h[0],
    pollution_predite_14h[0]
]


plt.figure(num="Évolution de la pollution durant la journée du 13/10/2024", figsize=(8, 5))
# Tracer les données de pollution avec des marqueurs
plt.plot(heures, pollution_values, marker='o', linestyle='-', color='b', label="Pollution prédite")


for i, txt in enumerate(pollution_values):
    plt.annotate(f'{txt:.1f} µg/m³', (heures[i], pollution_values[i]), textcoords="offset points", xytext=(0, 5), ha='center')


coefficients = np.polyfit(range(len(heures)), pollution_values, 1)
tendance = np.polyval(coefficients, range(len(heures)))
plt.plot(heures, tendance, color='r', linestyle='--', label="Ligne de tendance")

# un seuil de qualité de l'air (par exemple, le seuil de pollution PM2.5 à 50 µg/m³)
plt.axhline(y=50, color='g', linestyle=':', label="Seuil acceptable (50 µg/m³)")


plt.title("Évolution de la pollution prédite avec analyses")
plt.xlabel("Heures")
plt.ylabel("Pollution prédite (µg/m³ de PM2.5)")
plt.grid(True)


plt.legend()

# Afficher le graphique
plt.show()
