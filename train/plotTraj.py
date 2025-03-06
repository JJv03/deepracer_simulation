import matplotlib.pyplot as plt
import pandas as pd
import os

# Directorio donde están las trayectorias
trajectory_dir = "trajectories"

# Crear la figura
plt.figure(figsize=(10, 10))

# Recorrer todos los archivos en el directorio
for filename in sorted(os.listdir(trajectory_dir)):
    if filename.startswith("trajectory") and filename.endswith(".csv"):
        filepath = os.path.join(trajectory_dir, filename)
        
        # Leer la trayectoria
        df = pd.read_csv(filepath)
        
        # Graficar la trayectoria
        plt.plot(df["x"], df["y"], label=filename)

# Estilizar el gráfico
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.title("Evolución del Aprendizaje del Coche")
plt.legend(loc="best", fontsize=8)  # Muestra el nombre de cada archivo
plt.grid(True)
plt.show()
