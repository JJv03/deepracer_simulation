#!/usr/bin/env python3

import os
import math
import cv2
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from deepracer_env import DeepRacerEnv
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

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

def visualizar_decision(obs, action, step, model, output_dir, showFMapFilter):
    """Guarda una imagen visualizando la acción y algunos pesos de la red."""

    # Convertir la imagen de channel-first a channel-last
    img = obs[0].transpose(1, 2, 0).copy()
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Escalar si es muy pequeña
    if img.shape[0] < 200:
        img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

    altura, ancho, _ = img.shape
    ancho_info = 150  # Ancho para la columna de texto

    # Crear imagen en blanco para la columna de texto
    info_panel = np.ones((altura, ancho_info, 3), dtype=np.uint8) * 30  # fondo oscuro
    font_color = (0, 255, 0)
    font_scale = 0.5
    line_height = 16

    # Escribir información de la acción
    cv2.putText(info_panel, f"Step: {step}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)
    cv2.putText(info_panel, f"Angulo: {action[0][0]:.3f}", (10, 25 + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)
    cv2.putText(info_panel, f"Velocidad: {action[0][1]:.3f}", (10, 25 + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)

    # Extraer pesos si es posible
    if hasattr(model.policy, "mlp_extractor"):
        try:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).float().to(model.device)
                latent_pi, _ = model.policy.mlp_extractor(model.policy.features_extractor(obs_tensor))
                weights = latent_pi.cpu().numpy().flatten()[:10]
                for i, w in enumerate(weights):
                    y = 25 + (3 + i) * line_height
                    cv2.putText(info_panel, f"W{i}: {w:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255), 1)
        except Exception as e:
            print(f"No se pudieron extraer pesos: {e}")

    # Concatenar imagen original con panel de información
    img_out = np.hstack((img, info_panel))

    # Guardar la imagen
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"step_{step:05d}.png")
    cv2.imwrite(out_file, img_out)
    print(f"Guardada imagen de análisis en: {out_file}")

    # === NUEVO: Visualizar feature maps y filtros ===
    if showFMapFilter:
        try:
            policy = model.policy
            policy.eval()

            obs_tensor = torch.tensor(obs).float().to(policy.device)

            # Hook para capturar la salida de la primera capa convolucional
            feature_maps = []

            def hook_fn(module, input, output):
                feature_maps.append(output.detach().cpu())

            # Registrar el hook en la primera capa CNN (ajusta si tu extractor es diferente)
            hook = policy.features_extractor.cnn[0].register_forward_hook(hook_fn)

            with torch.no_grad():
                _ = policy(obs_tensor)

            hook.remove()

            fmap = feature_maps[0].squeeze(0)  # [num_filters, H, W]
            n_filters = fmap.shape[0]
            rows = int(n_filters ** 0.5)
            cols = (n_filters + rows - 1) // rows

            # Graficar feature maps
            plt.figure(figsize=(12, 12))
            for i in range(n_filters):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(fmap[i], cmap='viridis')
                plt.axis('off')
            plt.suptitle("Feature map", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"feature_map_{step:05d}.png"))
            plt.close()

            # Visualizar filtros (pesos de la primera capa)
            filters = policy.features_extractor.cnn[0].weight.data.cpu()
            filters = filters[:, 0, :, :]  # Primer canal si es imagen en escala de grises

            plt.figure(figsize=(12, 12))
            for i in range(filters.shape[0]):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(filters[i], cmap='viridis')
                plt.axis('off')
            plt.suptitle("Filters", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"filters_{step:05d}.png"))
            plt.close()

            print(f"Guardado feature map y filtros para step {step}")
        except Exception as e:
            print(f"No se pudo generar feature map: {e}")

def analizar_estructura_cnn(model):
    print("\nAnálisis de la red convolucional (CNN):\n")
    cnn = model.policy.features_extractor.cnn
    for i, layer in enumerate(cnn):
        if isinstance(layer, torch.nn.Conv2d):
            print(f"Capa {i}: Conv2D")
            print(f"   - Filtros de salida: {layer.out_channels}")
            print(f"   - Canales de entrada: {layer.in_channels}")
            print(f"   - Tamaño del kernel: {layer.kernel_size}")
            print(f"   - Stride: {layer.stride}")
            print(f"   - Shape de los pesos: {layer.weight.data.shape}")
            print()

def main():
    dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
    step = 1
    output_dir = os.path.expanduser('~/analisis')
    waypoints, thickness, long = extract_waypoints(dae_file, step)

    # Configurar rutas
    base_path = os.path.expanduser('~/models/deepracer_eval')
    save_path = os.path.join(base_path, 'best_model.zip')

    # base_path = os.path.expanduser('~/models')
    # save_path = os.path.join(base_path, 'deepracer_model.zip')

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
    analizar_estructura_cnn(model)
    
    obs = env.reset()
    done = False
    step_counter = 0
    showFMapFilter = False

    input("Presiona Enter para continuar...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print("Ang:", action[:, 0], "Vel:", action[:, 1])
        # action[:,0] = 0
        # action[:,1] = 0
        obs, reward, done, info = env.step(action)
        print("Rew:", reward)

        # Mostrar la imagen que ve el robot
        frame = obs[0].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
        cv2.imshow("Vista del Robot", frame)
        cv2.waitKey(1)  # Muestra la imagen durante al menos 1 ms

        step_counter += 1
        if step_counter % 100 == 0:
            visualizar_decision(obs, action, step_counter, model, output_dir, showFMapFilter)

    print("Ejecución finalizada.")
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
