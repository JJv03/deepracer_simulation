#!/usr/bin/env python3
# Importaciones necesarias
import os
import numpy as np
from scipy.spatial import KDTree
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from imitation.data import rollout
from deepracer_env import DeepRacerEnv
from train import extract_waypoints
from imitation.algorithms import bc
from stable_baselines3 import PPO

# Definición de la política experta
class ExpertPolicy:
    def __init__(self, waypoints, env):
        self.waypoints = np.array(waypoints)[:, :2]  # Guardamos solo las coordenadas (x,y)
        self.env = env
        self.kd_tree = KDTree(self.waypoints)  # KDTree para búsqueda rápida del waypoint más cercano
        
    def predict(self, observation):
        # Maneja predicción para batch de observaciones o una sola observación
        if observation.ndim == 2:
            # Observaciones en batch
            return np.array([self._predict_single(obs) for obs in observation])
        else:
            # Observación única
            return np.array([self._predict_single(observation)])

    def _predict_single(self, observation):
        # Obtener posición y orientación del coche
        model_position, model_orientation = self.env.get_model_state()
        robot_pos = model_position[:2]  # Solo x, y

        # Buscar el waypoint más cercano usando KDTree
        _, nearest_index = self.kd_tree.query(robot_pos)
        nearest_waypoint = self.waypoints[nearest_index]
        
        # Definir el siguiente waypoint al que debe dirigirse
        next_index = (nearest_index + 1) % len(self.waypoints)
        next_waypoint = self.waypoints[next_index]
        
        # Calcular vector de dirección hacia el siguiente waypoint
        direction_vector = np.array(next_waypoint) - np.array(robot_pos)
        direction_vector /= np.linalg.norm(direction_vector)  # Normalizar vector
        
        # Calcular el ángulo de orientación del coche a partir del quaternion
        x, y, z, w = model_orientation
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Vector que representa la dirección del coche
        car_vector = np.array([np.cos(yaw), np.sin(yaw)])

        # Calcular error angular (diferencia entre dirección actual y deseada)
        angle_error = np.arctan2(direction_vector[1], direction_vector[0]) - yaw
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi  # Normalizar entre [-pi, pi]

        # Decidir control: giro (steering) y aceleración (throttle)
        steering = np.clip(angle_error, -1.0, 1.0)  # Limitar el giro
        throttle = 1.0  # Mantener velocidad constante

        return np.array([steering, throttle])
        
    def __call__(self, obs, state=None, dones=None):
        # Método requerido para que sea compatible como política
        actions = self.predict(obs)
        return actions, state

# -------------------------------------------------------------------
# CONFIGURACIÓN DEL ENTORNO Y ENTRENAMIENTO

# Archivo de entorno (.dae) y parámetros
dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
step = 1  # Resolución de waypoints

# Extraer waypoints del archivo .dae
waypoints, thickness, long = extract_waypoints(dae_file, step)

# Crear entorno vectorizado para entrenamiento
venv = DeepRacerEnv(waypoints, thickness, long)     # Crea el entorno de simulación DeepRacer
venv = DummyVecEnv([lambda: Monitor(venv)])         # Envuelve el entorno en Monitor y lo vectoriza
venv = VecTransposeImage(venv)                      # Ajusta el formato de imágenes para redes neuronales convolucionales

# Crear la política experta basada en los waypoints
expert = ExpertPolicy(waypoints, venv.envs[0])

# Semilla para reproducibilidad
rng = np.random.default_rng()

# Número de pasos de demostración a recoger
n_timesteps = 3500

# Recopilar transiciones (observaciones + acciones) usando el experto
transitions = rollout.generate_transitions(
    policy=expert,
    venv=venv,
    n_timesteps=n_timesteps, 
    rng=rng
)

base_path = os.path.expanduser('~/models')
os.makedirs(base_path, exist_ok=True)
save_path = os.path.join(base_path, 'bc_deepracer_expert')

# Inicializar un agente PPO (aunque se usará solo su arquitectura de red)
policy = PPO('CnnPolicy', venv, verbose=1)

# Configurar Behavioral Cloning (BC) usando las demostraciones
bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    policy=policy.policy,  # Usamos la red neuronal de PPO
    rng=rng
)

# Entrenar el modelo para imitar al experto
bc_trainer.train(n_epochs=100)  # Número de épocas de entrenamiento (ajustable)

# Guardar el modelo entrenado
policy.save(save_path)
