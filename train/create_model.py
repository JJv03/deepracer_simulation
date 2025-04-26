#!/usr/bin/env python3
import numpy as np
from scipy.spatial import KDTree
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from deepracer_env import DeepRacerEnv
from train import extract_waypoints
from imitation.algorithms import bc
from stable_baselines3 import PPO

class ExpertPolicy:
    def __init__(self, waypoints, env):
        self.waypoints = np.array(waypoints)[:, :2]
        self.env = env
        self.kd_tree = KDTree(self.waypoints)
        
    def predict(self, observation):
        if observation.ndim == 2:
            # Batched observations
            return np.array([self._predict_single(obs) for obs in observation])
        else:
            # Single observation
            return np.array([self._predict_single(observation)])

    def _predict_single(self, observation):
        # Obtener la posición actual del robot
        model_position, model_orientation = self.env.get_model_state()
        robot_pos = model_position[:2]

        # Buscar el waypoint más cercano
        _, nearest_index = self.kd_tree.query(robot_pos)
        nearest_waypoint = self.waypoints[nearest_index]
        
        # Siguiente waypoint al que apuntar
        next_index = (nearest_index + 1) % len(self.waypoints)
        next_waypoint = self.waypoints[next_index]
        
        # Calcular el vector del coche a la siguiente waypoint
        direction_vector = np.array(next_waypoint) - np.array(robot_pos)
        direction_vector /= np.linalg.norm(direction_vector)
        
        # Calcular el ángulo del coche (a partir de la orientación en quaternion)
        x, y, z, w = model_orientation
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Vector de dirección del coche
        car_vector = np.array([np.cos(yaw), np.sin(yaw)])

        # Ángulo entre la dirección actual y hacia el siguiente waypoint
        angle_error = np.arctan2(direction_vector[1], direction_vector[0]) - yaw
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi  # Normalizar entre [-pi, pi]

        # Control simple:
        steering = np.clip(angle_error, -1.0, 1.0)  # Limitamos el giro entre -1 y 1
        throttle = 1.0  # Velocidad constante (puedes hacerla variable si quieres)

        return np.array([steering, throttle])
        

    def __call__(self, obs, state=None, dones=None):
        actions = self.predict(obs)
        return actions, state

# Env vectorizado

dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
step = 1
waypoints, thickness, long = extract_waypoints(dae_file, step)

venv = DummyVecEnv([lambda: DeepRacerEnv(waypoints, thickness, long)])

# Crear experto
expert = ExpertPolicy(waypoints, venv.envs[0])

rng = np.random.default_rng()
n_timesteps = 3500
# Recoger trajectories
transitions = rollout.generate_transitions(
    policy=expert,
    venv=venv,
    n_timesteps=n_timesteps, 
    rng=rng
)

# Entrenar desde cero (o inicializar desde algún modelo base si quieres)
policy = PPO('CnnPolicy', venv, verbose=1)

# Entrenamiento por Behavioral Cloning
bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    policy=policy.policy,
    rng=rng
)

bc_trainer.train(n_epochs=10)  # Ajusta epochs según resultados

# Guardar el modelo
policy.save("bc_deepracer_expert")
