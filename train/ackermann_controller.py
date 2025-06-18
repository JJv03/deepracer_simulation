#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from ackermann_msgs.msg import AckermannDriveStamped
from PyQt5 import QtWidgets, QtCore
import sys
from train import extract_waypoints
import numpy as np
from scipy.spatial import KDTree
import os
import cv2
import torch
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image as sensor_image
from cv_bridge import CvBridge, CvBridgeError

def visualizar_decision(obs, action, step, reward, output_dir):
    """Guarda una imagen visualizando la acción y algunos pesos de la red."""

    # Verifica las dimensiones del input
    if isinstance(obs, np.ndarray):
        img_data = obs
    else:
        img_data = np.array(obs)

    if len(img_data.shape) == 4:
        # Batch of images, pick the first
        img = img_data[0].transpose(1, 2, 0).copy()
    elif len(img_data.shape) == 3:
        # Single image, possibly already channel-last
        if img_data.shape[0] <= 3:  # Channels-first
            img = img_data.transpose(1, 2, 0).copy()
        else:
            img = img_data.copy()  # Assume it's already HWC
    elif len(img_data.shape) == 2:
        # Grayscale image, convert to 3-channel
        img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Formato de imagen inesperado: shape={img_data.shape}")

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

    # Concatenar imagen original con panel de información
    img_out = np.hstack((img, info_panel))

    # Guardar la imagen
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{step:05d}_step.png")
    cv2.imwrite(out_file, img_out)
    print(f"Guardada imagen de análisis en: {out_file}")

class AckermannGUI(QtWidgets.QWidget):
    def __init__(self):
        super(AckermannGUI, self).__init__()

        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, queue_size=10)
        rospy.init_node('ackermann_controller_gui', anonymous=True)

        self.setWindowTitle("Ackermann Controller")

        self.bridge = CvBridge()
        self.latest_img = None
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.output_dir = os.path.expanduser('~/analisis')

        # Velocidad y Ángulo iniciales
        self.speed = 0.0
        self.steering = 0.0

        self.model_position = [0.0, 0.0, 0.0]
        self.model_orientation = [0.0, 0.0, 0.0, 1.0]

        self.steps = 0
        self.max_steps = 10000

        dae_file = "/home/jvalle/robot_ws/src/deepracer_simulation/meshes/2022_april_open/2022_april_open.dae"
        step = 1
        waypoints, self.thickness, self.long = extract_waypoints(dae_file, step)
        self.waypoints = np.array(waypoints)[:, :2]
        self.kd_tree = KDTree(self.waypoints)

        _, nearest_index = self.kd_tree.query([-0.5456519086166459, -3.060323716659117])
        self.prevWaypoint = nearest_index

        self.numWaypoints = 0
        self.distance = 4

        self.initial_position = np.array([-0.5456519086166459, -3.060323716659117, -5.581931699989023e-06])  # x, y, z
        self.initial_orientation = np.array([6.1710519125244906e-06, 2.4181460708256078e-05, -0.2583623974492632, 0.9660480686598593])  # x, y, z, w (cuaternión)
        
        self.prevPos = [0, 0]
        self.prevPos[0] = self.initial_position[0]
        self.prevPos[1] = self.initial_position[1]

        self.weightProx = 2.5

        # Etiquetas para mostrar valores actuales
        self.speed_label = QtWidgets.QLabel(f"Speed: {self.speed:.1f} m/s")
        self.steering_label = QtWidgets.QLabel(f"Steering Angle: {self.steering:.2f} rad")
        self.position_label = QtWidgets.QLabel("Position: N/A")

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.speed_label)
        layout.addWidget(self.steering_label)
        layout.addWidget(self.position_label)
        self.setLayout(layout)

        # Timer para publicar los valores de velocidad y dirección
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_cmd)
        self.timer.start(100)  # Publicar cada 100 ms (10Hz)

        # Suscripción a la posición del modelo en Gazebo
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)

    def callback_image(self, data):
        bridge = CvBridge()
        self.image = cv2.resize(bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'), (160, 120))

    def model_state_callback(self, msg):
        # Obtener la posición del vehículo en la simulación
        vehicle_index = msg.name.index("racecar")
        position = msg.pose[vehicle_index].position
        orientation = msg.pose[vehicle_index].orientation

        self.model_position = [position.x, position.y, position.z]
        self.model_orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
        self.update_position_label(position)

    def update_position_label(self, position):
        # Actualizar la etiqueta de la posición
        self.position_label.setText(f"Position: x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}")

    def keyPressEvent(self, event):
        """Captura las teclas presionadas y realiza las acciones correspondientes"""
        if event.key() == QtCore.Qt.Key_W:  # Aumentar velocidad
            self.increase_speed()
        elif event.key() == QtCore.Qt.Key_S:  # Disminuir velocidad
            self.decrease_speed()
        elif event.key() == QtCore.Qt.Key_A:  # Girar a la izquierda
            self.turn_left()
        elif event.key() == QtCore.Qt.Key_D:  # Girar a la derecha
            self.turn_right()
        
    def increase_speed(self):
        if self.speed < 5.0:  # Limitar la velocidad máxima a 5 m/s
            self.speed += 0.5
        self.update_labels()

    def decrease_speed(self):
        if self.speed > 0.0:  # Limitar la velocidad mínima a 0 m/s
            self.speed -= 0.5
        self.update_labels()

    def turn_left(self):
        if self.steering > -1.0:  # Limitar el ángulo mínimo a -1.0 rad
            self.steering -= 0.1
        self.steering = max(self.steering, -1.0)  # Asegurarse que el ángulo no sea menor a -1.0 rad
        self.update_labels()

    def turn_right(self):
        if self.steering < 1.0:  # Limitar el ángulo máximo a 1.0 rad
            self.steering += 0.1
        self.steering = min(self.steering, 1.0)  # Asegurarse que el ángulo no sea mayor a 1.0 rad
        self.update_labels()

    def update_labels(self):
        # Actualizar las etiquetas con los valores actuales de velocidad y ángulo
        self.speed_label.setText(f"Speed: {self.speed:.1f} m/s")
        self.steering_label.setText(f"Steering Angle: {self.steering:.2f} rad")

    def publish_cmd(self):
        # Publicar el comando de velocidad y ángulo en el tópico
        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering
        self.pub.publish(msg)

        self.steps += 1
        reward, done, _, _, _ = self.reward_func()
        print(f"Step: {self.steps} | Reward: {reward:.3f} | Done: {done}")

        action = [[self.steering, self.speed]]
        visualizar_decision(
            self.image, action, self.steps, reward, self.output_dir
        )

    def reward_func(self):
        """
        Calculates reward based on:
        1. Track center adherence
        2. Carrot-on-stick approach using hyperbolic reward
        3. Progressive waypoint completion
        """
        total_reward = 0
        speed = self.speed
        if len(self.waypoints) < 2:
            return -1, True, 0.0, speed, False  # Not enough waypoints to calculate direction
        
        if speed < 0:
            print("VELOCIDAD")
            return -1000, True, 0.0, speed, False  # Negative speed (reverse) is incorrect behavior

        # Get current robot position
        robot_pos = self.model_position[:2]  # Only use x, y coordinates

        # Find nearest waypoint using KD-Tree
        _, nearest_index = self.kd_tree.query(robot_pos)
        nearest_waypoint = self.waypoints[nearest_index]

        next_index = (nearest_index + 1) % len(self.waypoints)  # Siguiente waypoint en el recorrido
        next_waypoint = self.waypoints[next_index]

        # Calcular el vector de dirección (normalizado)
        direction_vector = np.array(next_waypoint) - np.array(nearest_waypoint)
        direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

        x, y, z, w = self.model_orientation
        # Calcular el ángulo de giro (Yaw) en el plano XY
        theta = np.arctan2(2 * (w * z + x * y), 1 - 2 * (x**2 + z**2))
        
        # Vector dirección en 2D
        car_vector = np.array([np.cos(theta), np.sin(theta)])
        
        # print("Direccion robot:", car_vector)

        # Check if out of bounds
        distance_to_center = np.linalg.norm(robot_pos - nearest_waypoint)
        max_distance = self.thickness / 2
        if distance_to_center > max_distance:
            print("FUERA DE PISTA")
            # return 0, True
            # return -(self.max_steps - self.steps), True
            return -2.5, True, distance_to_center, speed, False
        
        # Calcular el coseno del ángulo entre el vector de dirección y el vector de orientación del robot
        orientation_reward  = np.dot(direction_vector_normalized, car_vector)

        if(orientation_reward  < 0.3):
            print("DIRECCIÓN CONTRARIA")
            # return 0, True
            # return -(self.max_steps - self.steps), True
            return -2.5, True, distance_to_center, speed, False

        # Calculate center reward (TANGENCIAL, ANTES HE HECO EL COSENO, PUES AHORA SENO)
        center_reward = 1.0 - (distance_to_center / max_distance)
        
        # Proximity reward (carrot-on-stick approach)
        next_index = (nearest_index + self.distance) % len(self.waypoints)
        next_waypoint = self.waypoints[next_index]
        
        # proximity_reward = np.linalg.norm(robot_pos - self.prevPos)

        robot_pos = np.array(robot_pos)
        self.prevPos = np.array(self.prevPos)
        moved = np.linalg.norm(robot_pos - self.prevPos)
        # assume max possible moved ~ e.g. throttle*dt*max_speed ~ 5*0.025*? ~= 0.125
        max_step = 0.045
        proximity_reward = min(moved / max_step, 1.0)

        self.prevPos = robot_pos

        # Hyperbolic reward function that increases as car gets closer to target
        
        # Combine rewards
        total_reward += (
            center_reward * 1 +         # Stay centered on track (increased weight)
            orientation_reward  * 1 +   # Correct orientation
            proximity_reward * 2          # Chase the carrot (target waypoint)
        )

        # print("Center reward:", center_reward * self.weightCenter)
        # print("Orientation reward:", orientation_reward * self.weightOrient)
        # print("Prox reward:", proximity_reward * self.weightProx)
        
        # if speed < 0.3:
        #    total_reward -= 5.0
        # elif speed < 1:
        #     total_reward -= 1.0 / speed
        # else:
        #     total_reward += 1.0

        # print("Total reward:", total_reward)

        return total_reward, False, distance_to_center, speed, False

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = AckermannGUI()
    gui.show()
    sys.exit(app.exec_())
