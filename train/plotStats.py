import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("/home/jvalle/models/deepracer_logs/actions_reward/training_log.csv")

# Asegurar que los datos están en el tipo correcto
df['step'] = pd.to_numeric(df['step'], errors='coerce')
df['action_wheel_angle'] = pd.to_numeric(df['action_wheel_angle'], errors='coerce')
df['action_speed'] = pd.to_numeric(df['action_speed'], errors='coerce')
df['reward'] = pd.to_numeric(df['reward'], errors='coerce')

# Eliminar filas con valores no numéricos o nulos
df.dropna(subset=['step', 'action_wheel_angle', 'action_speed', 'reward'], inplace=True)

# Crear una figura y subplots
plt.figure(figsize=(12, 8))

# Graficar el ángulo del volante
plt.subplot(3, 1, 1)
plt.plot(df['step'].values, df['action_wheel_angle'].values, label='Wheel Angle', color='blue')
plt.ylabel('Wheel Angle')
plt.title('Training Log Over Steps')
plt.grid(True)
plt.legend()

# Graficar la velocidad
plt.subplot(3, 1, 2)
plt.plot(df['step'].values, df['action_speed'].values, label='Speed', color='green')
plt.ylabel('Speed')
plt.grid(True)
plt.legend()

# Graficar la recompensa
plt.subplot(3, 1, 3)
plt.plot(df['step'].values, df['reward'].values, label='Reward', color='orange')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()

# Ajustar el layout
plt.tight_layout()
plt.show()
