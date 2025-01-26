#!/usr/bin/env python3

import math
import os
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from deepracer_env import DeepRacerEnv

import xml.etree.ElementTree as ET

# Callback para detener el entrenamiento después de un número fijo de episodios
class EarlyStopWithDisplayCallback(BaseCallback):
    def __init__(self, env, max_episodes=5, verbose=0):
        super(EarlyStopWithDisplayCallback, self).__init__(verbose)
        self.env = env
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self):
        # Mostrar imagen de la cámara
        image = self.env.image
        if image is not None:
            cv2.imshow('Vista del Robot', image)
            cv2.waitKey(1)  # Actualiza la ventana
        
        reward = self.locals["rewards"][-1]
        print(f"Paso: {self.num_timesteps}, Recompensa: {reward}")

        # Contar episodios
        if self.locals['dones'][0]:  # Un episodio terminó
            self.episode_count += 1
            print(f"------Episodio {self.episode_count} terminado------")

        # Detener entrenamiento tras max_episodes
        return self.episode_count < self.max_episodes


def extract_waypoints(dae_file, step=1):
    """
    Extrae waypoints desde un archivo .dae y los guarda en una lista.
    Cada cuarto waypoint representa un rectángulo, por lo que el punto
    de cada cuatro waypoints consecutivos será el centro del rectángulo.

    :param dae_file: Ruta al archivo .dae
    :param array_id: ID del array que contiene las coordenadas (por defecto, 'cl-verts-array')
    :param step: Intervalo para seleccionar waypoints (por defecto, 1)
    :return: Lista de waypoints
    """
    # Cargar y analizar el archivo .dae
    tree = ET.parse(dae_file)
    root = tree.getroot()

    array_id='cl-verts-array'
    
    # Buscar el array de coordenadas con el ID proporcionado
    waypoints = []
    for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
        if float_array.attrib.get('id') == array_id:
            data = list(map(float, float_array.text.strip().split()))

            # Las coordenadas vienen en grupos de tres (x, y, z)
            for i in range(0, len(data), 12 * step):  # Cada 12 valores representan 4 waypoints (4 * 3)
                # Extraer los 4 puntos (12 valores: 4 * 3 = 12)
                if i + 11 < len(data):  # Asegurarse de que hay 4 puntos
                    points = [
                        data[i:i+3],        # Primer waypoint
                        data[i+3:i+6],      # Segundo waypoint
                        data[i+6:i+9],      # Tercer waypoint
                        data[i+9:i+12]      # Cuarto waypoint
                    ]
                    # Calcular el centro del rectángulo (promedio de los 4 puntos)
                    center = [
                        sum([point[0] for point in points]) / 4,  # Promedio de las X
                        sum([point[1] for point in points]) / 4,  # Promedio de las Y
                        sum([point[2] for point in points]) / 4   # Promedio de las Z
                    ]
                    waypoints.append(center)

            break
    
    ol_waypoint = []
    for array_id in ['ol-verts-array']:  # Ahora solo buscamos el primer de 'ol-verts-array'
        for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
            if float_array.attrib.get('id') == array_id:
                data = list(map(float, float_array.text.strip().split()))
                # Solo tomamos el primer conjunto de 3 valores (x, y, z)
                if len(data) >= 3:
                    ol_waypoint.append(data[:3])  # Solo el primer waypoint
                break  # Salimos después de tomar el primer waypoint

    # Finalmente, extraer el primer waypoint de 'il-verts-array'
    il_waypoint = []
    for array_id in ['il-verts-array']:  # Ahora solo buscamos el primer de 'il-verts-array'
        for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
            if float_array.attrib.get('id') == array_id:
                data = list(map(float, float_array.text.strip().split()))
                # Solo tomamos el primer conjunto de 3 valores (x, y, z)
                if len(data) >= 3:
                    il_waypoint.append(data[:3])  # Solo el primer waypoint
                break  # Salimos después de tomar el primer waypoint
            
    # Calcular la distancia entre los puntos 'il_waypoint' y 'ol_waypoint'
    if il_waypoint and ol_waypoint:
        # Fórmula de distancia euclidiana en 3D
        distance = math.sqrt((ol_waypoint[0][0] - il_waypoint[0][0]) ** 2 + 
                             (ol_waypoint[0][1] - il_waypoint[0][1]) ** 2 + 
                             (ol_waypoint[0][2] - il_waypoint[0][2]) ** 2)
        print(f"Distancia entre 'il' y 'ol': {distance} unidades")
    else:
        print("No se encontraron los puntos 'il' o 'ol' para calcular la distancia.")

    print(f"Waypoints extraídos: {len(waypoints)} puntos")
    print(waypoints)
    return waypoints, distance

def main():
    # Configuración de los waypoints
    dae_file = "/home/arob/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"  # Archivo proporcionado
    # "cl-verts-array" (linea central) - "road-verts-array" (Carretera) - "ol-verts-array" (Linea exterior) - "il-verts-array" (Linea interior)
    step = 1  # Intervalo para seleccionar waypoints

    # Ejecutar la extracción (valores organizados en grupos de tres, 5940/3=1980 puntos)
    waypoints, thickness = extract_waypoints(dae_file, step)

    # Crear el entorno
    env = DeepRacerEnv(waypoints, thickness)

    # Configurar rutas
    base_path = os.path.expanduser('~/models')
    logs_path = os.path.join(base_path, 'deepracer_logs')
    save_path = os.path.join(base_path, 'deepracer_model')

    # Crear directorios si no existen
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    # Configurar el modelo
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=logs_path)

    # Entrenar con el callback
    callback = EarlyStopWithDisplayCallback(env, max_episodes=1)
    model.learn(total_timesteps=10000, callback=callback)

    # Guardar el modelo
    try:
        model.save(save_path)
        print(f"Modelo guardado exitosamente en {save_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

    # Cerrar ventanas y entorno
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()