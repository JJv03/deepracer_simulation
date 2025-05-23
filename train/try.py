#!/usr/bin/env python3

import os
import math
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from deepracer_env import DeepRacerEnv
import xml.etree.ElementTree as ET

def calcular_distancia(punto1, punto2):
    return math.sqrt((punto2[0] - punto1[0])**2 + (punto2[1] - punto1[1])**2 + (punto2[2] - punto1[2])**2)

def extract_waypoints(dae_file, step=1):
    """
    Extrae waypoints desde un archivo .dae y los guarda en una lista.
    Calcula también la distancia entre la línea interior (il) y la exterior (ol).
    """
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
                    center = [
                        sum([point[0] for point in points]) / 4,
                        sum([point[1] for point in points]) / 4,
                        sum([point[2] for point in points]) / 4
                    ]
                    waypoints.append(center)
            break

    # Extraer puntos 'ol-verts-array' y 'il-verts-array' para calcular el grosor de la pista
    ol_waypoint, il_waypoint = [], []
    for array_id in ['ol-verts-array', 'il-verts-array']:
        for float_array in root.iter('{http://www.collada.org/2005/11/COLLADASchema}float_array'):
            if float_array.attrib.get('id') == array_id:
                data = list(map(float, float_array.text.strip().split()))
                if len(data) >= 3:
                    if array_id == 'ol-verts-array':
                        ol_waypoint.append(data[:3])
                    elif array_id == 'il-verts-array':
                        il_waypoint.append(data[:3])
                break

    if il_waypoint and ol_waypoint:
        distance = math.sqrt((ol_waypoint[0][0] - il_waypoint[0][0]) ** 2 +
                             (ol_waypoint[0][1] - il_waypoint[0][1]) ** 2 +
                             (ol_waypoint[0][2] - il_waypoint[0][2]) ** 2)
        print(f"Distancia entre 'il' y 'ol': {distance} m")
    else:
        print("No se encontraron los puntos 'il' o 'ol' para calcular la distancia.")
        distance = 0.0

    print(f"Waypoints extraídos: {len(waypoints)} puntos")
    
    long = 0.0
    for i in range(1, len(waypoints)):
        long += calcular_distancia(waypoints[i-1], waypoints[i])
        
    print(f"Longitud de la pista: {long} m")
    
    return waypoints, distance, long

def main():
    dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
    step = 1
    waypoints, thickness, long = extract_waypoints(dae_file, step)

    # Configurar rutas
    # base_path = os.path.expanduser('~/models/deepracer_eval')
    # save_path = os.path.join(base_path, 'best_model.zip')

    base_path = os.path.expanduser('~/models')
    save_path = os.path.join(base_path, 'deepracer_model.zip')

    # base_path = os.path.expanduser('~/models')
    # save_path = os.path.join(base_path, 'bc_deepracer_expert.zip')
    
    if not os.path.exists(save_path):
        print(f"No se encontró un modelo entrenado en {save_path}")
        return

    # Crear el entorno de simulación
    env = DeepRacerEnv(waypoints, thickness, long)
    env = DummyVecEnv([lambda: Monitor(env)])
    env = VecTransposeImage(env)
    
    # Cargar el modelo entrenado
    model = PPO.load(save_path)
    print("Modelo cargado exitosamente.")
    
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # action[:,1] = action[:,1]*5
        print ("Ang:", action[:,0], "Vel:", action[:,1])
        obs, reward, done, info = env.step(action)
        print("Rew:", reward)
    
    print("Ejecución finalizada.")
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
