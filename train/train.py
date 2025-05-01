#!/usr/bin/env python3

import os
import math
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from deepracer_env import DeepRacerEnv
import xml.etree.ElementTree as ET
import torch
import csv
import numpy as np

class CSVLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        base_path = os.path.expanduser('~/models')
        logs_path = os.path.join(base_path, 'deepracer_logs')
        log_dir = os.path.join(logs_path, 'actions_reward')
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = os.path.join(log_dir, 'training_log.csv')

        # Crear archivo y escribir encabezado si no existe
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["step", "action_wheel_angle", "action_speed", "reward"])

    def _on_step(self) -> bool:
        step = self.num_timesteps  # Número de pasos de entrenamiento
        actions = self.locals["actions"]
        rewards = self.locals["rewards"]

        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            for i, (action, reward) in enumerate(zip(actions, rewards)):
                writer.writerow([step, np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 5.0), reward])

        return True

    def _on_training_end(self) -> None:
        print(f"Training log saved to {self.csv_file}")


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
    print(torch.cuda.is_available())  # Debería imprimir True
    print(torch.cuda.device_count())  # Número de GPUs detectadas (al menos 1)
    print(torch.cuda.get_device_name(0))  # Nombre de la GPU

    # Configuración de los waypoints
    dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
    step = 1
    waypoints, thickness, long = extract_waypoints(dae_file, step)

    # Crear el entorno y monitor
    env = DeepRacerEnv(waypoints, thickness, long)    # Crea el entorno de simulación DeepRacer
    env = DummyVecEnv([lambda: Monitor(env)])   # Envuelve el entorno en Monitor y lo vectoriza
    env = VecTransposeImage(env)                # Ajusta el formato de imágenes para redes neuronales convolucionales


    # Configurar rutas
    base_path = os.path.expanduser('~/models')
    logs_path = os.path.join(base_path, 'deepracer_logs')
    save_path = os.path.join(base_path, 'deepracer_model')
    eval_path = os.path.join(base_path, 'deepracer_eval')
    # model_path = os.path.join(base_path, 'bc_deepracer_expert.zip')
    model_path = os.path.join(base_path, 'base_model.zip')

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    # Configurar el modelo con hiperparámetros ajustados
    model = PPO(
        "CnnPolicy", env, verbose=1, tensorboard_log=logs_path,
        learning_rate=2.5e-4, gamma=0.99, n_steps=2048, batch_size=128, clip_range=0.25,
        device="cuda"
    )

    if os.path.exists(model_path):
        print(f"Cargando pesos del modelo preentrenado desde {model_path}...")
        pretrained_model = PPO.load(model_path, device="cuda", weights_only=True)
        model.policy.load_state_dict(pretrained_model.policy.state_dict())
        print("Pesos cargados exitosamente.")
    else:
        print("No se encontró modelo preentrenado, entrenando desde cero.")

    # Callbacks para evaluar y guardar el modelo
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=save_path, name_prefix="deepracer_checkpoint")
    
    eval_callback = EvalCallback(env, best_model_save_path=eval_path, log_path=eval_path, eval_freq=5000, deterministic=True, render=False)

    csv_callback = CSVLoggingCallback()
    print(f"Entrenando en: {model.policy.device}")
    # Entrenar el modelo
    try:
        print("Comenzando el entrenamiento...")
        model.learn(total_timesteps=50000, callback=[checkpoint_callback, eval_callback, csv_callback])
        model.save(save_path)
        print(f"Modelo guardado exitosamente en {save_path}")
        print("Entrenamiento finalizado")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")

    # Cerrar el entorno
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()