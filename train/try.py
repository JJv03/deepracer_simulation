#!/usr/bin/env python3

import os
import math
import time
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

def extract_waypoints(dae_file, step=1, reverse=False):
    """
    Extrae waypoints desde un archivo .dae y los guarda en una lista.
    Calcula también la distancia entre la línea interior (il) y la exterior (ol).

    Parámetros:
    - dae_file: ruta al archivo .dae
    - step: salto para tomar waypoints (por defecto 1)
    - reverse: si es True, los waypoints se guardan en orden inverso
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

    if reverse:
        waypoints = waypoints[::-1]

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

def visualizar_decision(obs, action, step, reward, model, output_dir, showFMapFilter):
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
    rew_color = (0, 165, 255)
    font_scale = 0.5
    line_height = 16

    # Escribir información de la acción
    cv2.putText(info_panel, f"Step: {step}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)
    cv2.putText(info_panel, f"Angulo: {action[0][0]:.3f}", (10, 25 + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)
    cv2.putText(info_panel, f"Velocidad: {action[0][1]:.3f}", (10, 25 + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1)
    cv2.putText(info_panel, f"Reward: {float(reward):.3f}", (10, 25 + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, rew_color, 1)
    
    # Extraer pesos si es posible
    if hasattr(model.policy, "mlp_extractor"):
        try:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).float().to(model.device)
                latent_pi, _ = model.policy.mlp_extractor(model.policy.features_extractor(obs_tensor))
                weights = latent_pi.cpu().numpy().flatten()[:10]
                for i, w in enumerate(weights):
                    y = 42 + (3 + i) * line_height
                    cv2.putText(info_panel, f"W{i}: {w:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255), 1)
        except Exception as e:
            print(f"No se pudieron extraer pesos: {e}")

    # Concatenar imagen original con panel de información
    img_out = np.hstack((img, info_panel))

    # Guardar la imagen
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{step:05d}_step.png")
    cv2.imwrite(out_file, img_out)
    print(f"Guardada imagen de análisis en: {out_file}")

    # === Visualizar feature maps y filtros de varias capas ===
    if showFMapFilter:
        try:
            policy = model.policy
            policy.eval()

            obs_tensor = torch.tensor(obs).float().to(policy.device)

            # Lista de capas CNN de las que quieres capturar feature maps
            layers_to_hook = [0, 2, 4]

            feature_maps = [[] for _ in layers_to_hook]
            hooks = []

            # Función generadora de hooks para capturar outputs y guardar con la capa correcta
            def make_hook(i):
                def hook_fn(module, input, output):
                    feature_maps[i].append(output.detach().cpu())
                return hook_fn

            # Registrar hooks para cada capa indicada
            for i, layer_idx in enumerate(layers_to_hook):
                hook = policy.features_extractor.cnn[layer_idx].register_forward_hook(make_hook(i))
                hooks.append(hook)

            with torch.no_grad():
                _ = policy(obs_tensor)

            # Quitar hooks
            for hook in hooks:
                hook.remove()

            # Visualizar feature maps y filtros para cada capa
            for i, layer_idx in enumerate(layers_to_hook):
                fmap = feature_maps[i][0].squeeze(0)  # [num_filters, H, W]
                n_filters = fmap.shape[0]
                rows = int(n_filters ** 0.5)
                cols = (n_filters + rows - 1) // rows

                # Feature maps
                plt.figure(figsize=(12, 12))
                for f in range(n_filters):
                    plt.subplot(rows, cols, f + 1)
                    plt.imshow(fmap[f], cmap='viridis')
                    plt.axis('off')
                    plt.title(f"Layer {layer_idx} - Filter {f}", fontsize=8)
                plt.suptitle(f"Feature maps - Layer {layer_idx}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{step:05d}_feature_map_layer{layer_idx}.png"))
                plt.close()

                # Filtros (pesos)
                filters = policy.features_extractor.cnn[layer_idx].weight.data.cpu()
                filters = filters[:, 0, :, :]  # Ajusta si tienes varias canales

                plt.figure(figsize=(12, 12))
                for f in range(filters.shape[0]):
                    plt.subplot(rows, cols, f + 1)
                    plt.imshow(filters[f], cmap='viridis')
                    plt.axis('off')
                    plt.title(f"Layer {layer_idx} - Filter {f}", fontsize=8)
                plt.suptitle(f"Filters - Layer {layer_idx}", fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{step:05d}_filters_layer{layer_idx}.png"))
                plt.close()

            print(f"Guardado feature maps y filtros para step {step}")

        except Exception as e:
            print(f"No se pudo generar feature maps: {e}")

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
    # dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_march_open/2022_march_open.dae"
    # dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_october_open/2022_october_open.dae"
    step = 1
    output_dir = os.path.expanduser('~/analisis')
    waypoints, thickness, long = extract_waypoints(dae_file, step)

    # Configurar rutas
    base_path = os.path.expanduser('~/models/deepracer_eval')
    # save_path = os.path.join(base_path, 'best_model.zip')
    save_path = os.path.join(base_path, 'all.zip')

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
    genImages = True
    showFMapFilter = False

    # === Inicializa acumuladores ===
    finished = False
    distancia_total = 0.0
    velocidad_total = 0.0
    tiempo_inicio = time.time()

    input("Presiona Enter para continuar...")
    while not done and not finished:
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

        # === Acumuladores para métricas ===
        if 'distance_from_center' in info[0]:
            distancia_total += abs(info[0]['distance_from_center'])  # en metros

        if 'speed' in info[0]:
            velocidad_total += info[0]['speed']  # en m/s

        step_counter += 1
        if step_counter % 100 == 0 and genImages:
            visualizar_decision(obs, action, step_counter, reward, model, output_dir, showFMapFilter)

        if 'finished' in info[0]:
            finished = info[0]['finished']
            if finished: env.reset()

    print("Ejecución finalizada.")

    tiempo_fin = time.time()
    duracion_total = tiempo_fin - tiempo_inicio

    # === Cálculos finales ===
    distancia_media = distancia_total / step_counter if step_counter else 0
    velocidad_media = velocidad_total / step_counter if step_counter else 0
    tiempo_medio_vuelta = duracion_total  # Solo una vuelta en este caso

    # === Reporte final ===
    print("\n--- Resultados de la vuelta ---")
    print(f"Distancia media al centro de la pista (m): {distancia_media:.3f}")
    print(f"Anchura de la pista (m): {thickness:.3f}")
    print(f"Velocidad media (m/s): {velocidad_media:.3f}")
    print(f"Tiempo total de vuelta (s): {tiempo_medio_vuelta:.2f}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
