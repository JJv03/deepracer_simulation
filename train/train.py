#!/usr/bin/env python3

import os
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from deepracer_env import DeepRacerEnv

import csv
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

def extract_waypoints_to_csv(dae_file, output_csv, array_id='road-verts-array', step=10):
    """
    Extrae waypoints desde un archivo .dae y los guarda en un archivo CSV.

    :param dae_file: Ruta al archivo .dae
    :param output_csv: Ruta al archivo de salida .csv
    :param array_id: ID del array que contiene las coordenadas (por defecto, 'road-verts-array')
    :param step: Intervalo para seleccionar waypoints (por defecto, 10)
    """
    # Cargar y analizar el archivo .dae
    tree = ET.parse(dae_file)
    root = tree.getroot()

    # Buscar el array de coordenadas con el ID proporcionado
    waypoints = []
    for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
        if float_array.attrib.get('id') == array_id:
            data = list(map(float, float_array.text.strip().split()))

            # Las coordenadas vienen en grupos de tres (x, y, z)
            for i in range(0, len(data), 3 * step):
                waypoint = data[i:i+3]  # Extraer x, y, z
                if len(waypoint) == 3:
                    waypoints.append(waypoint)
            break

    # Guardar los waypoints en un archivo CSV
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['x', 'y', 'z'])  # Encabezados
        writer.writerows(waypoints)  # Escribir las filas de waypoints

    print(f"Waypoints extraídos y guardados en {output_csv}")

def main():
    # Configuración de los waypoints
    dae_file = "../meshes/2022_april_open/2022_april_open.dae"  # Archivo proporcionado
    output_csv = "waypoints.csv"  # Archivo de salida
    array_id = "cl-verts-array"  # "cl-verts-array" (linea central) - "road-verts-array" (Carretera) - "ol-verts-array" (Linea exterior)
    step = 1  # Intervalo para seleccionar waypoints

    # Ejecutar la extracción (valores organizados en grupos de tres, 5940/3=1980 puntos)
    extract_waypoints_to_csv(dae_file, output_csv, array_id, step)

    # Crear el entorno
    env = DeepRacerEnv()

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