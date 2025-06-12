import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("/home/jvalle/models/deepracer_logs/actions_reward/training_log.csv")

# Asegurar que los datos están en el tipo correcto
df['step'] = pd.to_numeric(df['step'], errors='coerce')
df['action_wheel_angle'] = pd.to_numeric(df['action_wheel_angle'], errors='coerce')
df['action_speed'] = pd.to_numeric(df['action_speed'], errors='coerce')
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
df['fps'] = pd.to_numeric(df['fps'], errors='coerce')

# Eliminar filas con valores no numéricos o nulos
df.dropna(subset=['step', 'action_wheel_angle', 'action_speed', 'reward', 'fps'], inplace=True)

# Crear una figura y subplots
plt.figure(figsize=(12, 10))

# Graficar el ángulo del volante
plt.subplot(4, 1, 1)
plt.plot(df['step'].values, df['action_wheel_angle'].values, label='Wheel Angle', color='blue')
plt.ylabel('Wheel Angle')
plt.title('Training Log Over Steps')
plt.grid(True)
plt.legend()

# Graficar la velocidad
plt.subplot(4, 1, 2)
plt.plot(df['step'].values, df['action_speed'].values, label='Speed', color='green')
plt.ylabel('Speed')
plt.grid(True)
plt.legend()

# Graficar la recompensa
plt.subplot(4, 1, 3)
plt.plot(df['step'].values, df['reward'].values, label='Reward', color='orange')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()

# Graficar los FPS
plt.subplot(4, 1, 4)
plt.plot(df['step'].values, df['fps'].values, label='FPS', color='purple')
plt.xlabel('Step')
plt.ylabel('FPS')
plt.grid(True)
plt.legend()

# Ajustar el layout
plt.tight_layout()
plt.show()
