import matplotlib.pyplot as plt
import pandas as pd
import os
import xml.etree.ElementTree as ET
import re

# Directorio donde están las trayectorias
trajectory_dir = os.path.expanduser('~/trajectories')

# Archivo DAE que contiene los waypoints
dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
# dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_march_open/2022_march_open.dae"
# dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_october_open/2022_october_open.dae"

# Paso para la lectura de waypoints
step = 1

# Parsear el archivo DAE para extraer los waypoints
tree = ET.parse(dae_file)
root = tree.getroot()
array_id = 'cl-verts-array'
waypoints = []

for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
    if float_array.attrib.get('id') == array_id:
        data = list(map(float, float_array.text.strip().split()))
        for i in range(0, len(data), 12 * step):
            if i + 11 < len(data):
                points = [
                    data[i:i+3], data[i+3:i+6], data[i+6:i+9], data[i+9:i+12]
                ]
                # Calcular el centro de los puntos de cada waypoint
                center = [
                    sum([point[0] for point in points]) / 4,
                    sum([point[1] for point in points]) / 4,
                    sum([point[2] for point in points]) / 4
                ]
                waypoints.append(center)
        break

print(f"Waypoints extraídos: {len(waypoints)} puntos")

# Crear la figura para graficar las trayectorias y los waypoints
plt.figure(figsize=(10, 10))

# Obtener la lista de archivos y extraer el número de episodio
files = sorted(
    [f for f in os.listdir(trajectory_dir) if f.startswith("trajectory") and f.endswith(".csv")], 
    key=lambda x: int(re.search(r'ep(\d+)', x).group(1)))

# Recorrer los archivos con un índice para manejar zorder
print(len(files), "trajectories")
# for i, filename in enumerate(files[-10:]):
for i, filename in enumerate(files):
    # if i > 10:
    #     break # Mostras las X trayectorias primeras 
    filepath = os.path.join(trajectory_dir, filename)
    
    # Leer la trayectoria
    df = pd.read_csv(filepath)
    
    # Asegurarse de que las columnas "x" y "y" sean válidas
    if df["x"].dtype in ['float64', 'int64'] and df["y"].dtype in ['float64', 'int64']:
        # Asignar un zorder mayor a trayectorias más recientes
        plt.plot(df["x"].values, df["y"].values, label=filename, zorder=len(files) - i)
    else:
        print(f"Advertencia: Datos no válidos en el archivo {filename}")

# Graficar los waypoints
waypoints_x = [wp[0] for wp in waypoints]
waypoints_y = [wp[1] for wp in waypoints]

# Graficar los waypoints con un estilo diferente (por ejemplo, círculos rojos)
plt.scatter(waypoints_x, waypoints_y, color='red', marker='o', label="Waypoints", zorder=0)

# Mostrar direcciones waypoints
for i in range(len(waypoints)):
    if(i % 6 == 0):
        if i + 6 < len(waypoints):
            x_start, y_start = waypoints[i][0], waypoints[i][1]
            x_end, y_end = waypoints[i + 6][0], waypoints[i + 6][1]
            plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.6)

# Estilizar el gráfico
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.title("Evolución del Aprendizaje del Coche")
plt.legend(loc="best", fontsize=8)  # Muestra el nombre de cada archivo y los waypoints
plt.grid(True)
plt.show()
