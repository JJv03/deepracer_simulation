#!/usr/bin/env python3

import os
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from deepracer_env import DeepRacerEnv

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
