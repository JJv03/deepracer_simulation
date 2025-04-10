import os
import pandas as pd
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
import rospy
from std_srvs.srv import Empty
from sensor_msgs.msg import Image as sensor_image
import cv2
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from scipy.spatial import KDTree

class DeepRacerEnv(gym.Env):
    def __init__(self, waypoints, thickness, long):
        super(DeepRacerEnv, self).__init__()
        self.steps = 0
        self.max_steps = 10000
        
        self.episode_count = 0  # Contador de episodios
        self.frecTraj = 5       # cada cuantas trayectorias se guarda
        self.positions = []     # Almacenar posiciones del coche
        
        self.ack_publisher = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, queue_size=100)
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_model_states)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.init_node('deepracer_rl', anonymous=True)
        
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.model_position = np.zeros(3)
        self.model_orientation = np.zeros(4)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        
        self.waypoints = np.array(waypoints)[:, :2]  # Solo tomar x, y
        self.thickness = thickness
        self.long = long
        self.kd_tree = KDTree(self.waypoints)  # Construcción del KD-Tree

        self.numWaypoints = 0
        _, nearest_index = self.kd_tree.query([-0.5456519086166459, -3.060323716659117])
        self.prevWaypoint = nearest_index
        self.distance = 4
        self.distanceBetweenWaypoints = np.linalg.norm(self.waypoints[nearest_index] - self.waypoints[(nearest_index+1)%len(self.waypoints)]) * self.distance
        
        self.times = 0.0
        
        self.speed = 0

        # Pesos
        self.weightProx = 2.5
        self.weightDir = 1
        self.weightSpeed = 0.6

        # Inicializar la posición inicial del robot
        self.initial_position = np.array([-0.5456519086166459, -3.060323716659117, -5.581931699989023e-06])  # x, y, z
        self.initial_orientation = np.array([6.1710519125244906e-06, 2.4181460708256078e-05, -0.2583623974492632, 0.9660480686598593])  # x, y, z, w (cuaternión)
        
        self.prevPos = [0, 0]
        self.prevPos[0] = self.initial_position[0]
        self.prevPos[1] = self.initial_position[1]

        base_path = os.path.expanduser('~/trajectories')
        os.makedirs(base_path, exist_ok=True)

    def reset(self, seed=None):
        super().reset(seed=seed)
        """
        Reinicia el entorno y devuelve un estado inicial.
        """
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        print("RESET!", self.episode_count)

        # Si el episodio es múltiplo de self.frecTraj y hay datos, guardamos la trayectoria
        if self.episode_count % self.frecTraj == 1 and self.positions:
            df = pd.DataFrame(self.positions, columns=["x", "y"])
            df.to_csv(f"~/trajectories/trajectory_ep{self.episode_count}.csv", index=False)
            print(f"Trayectoria del episodio {self.episode_count} guardada en CSV.")
        
        # Reiniciar la lista de posiciones para el nuevo episodio
        self.positions = []
        self.episode_count += 1  # Incrementar el número de episodios
        
        self.reward = None
        self.state = None
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.steps = 0

        _, nearest_index = self.kd_tree.query([-0.5456519086166459, -3.060323716659117])
        self.prevWaypoint = nearest_index
        self.numWaypoints = 0

        # Reiniciar posición en Gazebo
        self.reset_model_state()
        
        self.times = 0.0

        # Inicializa el estado con valores aleatorios en los rangos definidos
        self.send_action(0, 0)  # Establece el acelerador a 0
        self.speed = 0
        return self.image, {}  # Devuelve el estado inicial y un diccionario vacío

    def reset_model_state(self):
        """
        Reinicia la posición y orientación del robot en Gazebo utilizando el servicio /gazebo/set_model_state.
        """
        try:
            rospy.wait_for_service('/gazebo/set_model_state')
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            
            model_state = ModelState()
            model_state.model_name = "racecar"  # El nombre del modelo en Gazebo
            model_state.pose.position.x = self.initial_position[0]
            model_state.pose.position.y = self.initial_position[1]
            model_state.pose.position.z = self.initial_position[2]
            model_state.pose.orientation.x = self.initial_orientation[0]
            model_state.pose.orientation.y = self.initial_orientation[1]
            model_state.pose.orientation.z = self.initial_orientation[2]
            model_state.pose.orientation.w = self.initial_orientation[3]
            
            # Establecer la nueva posición en Gazebo
            set_model_state(model_state)
        except rospy.ServiceException as e:
            rospy.logerr("Failed to set model state: %s", e)

    def step(self, action):
        """
        Aplica una acción y avanza un paso en el entorno.
        """

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # ----------------------------------------- #
        # Primera aproximación a done por distancia #
        # ----------------------------------------- #
        
        # current_position = self.model_position[:2]  # Solo considerar x, y
        # distance_step = np.linalg.norm(np.array(current_position) - np.array(self.prev_position))  # Distancia recorrida en este paso
        # self.distance_traveled += distance_step  # Sumar a la distancia total recorrida
        # self.prev_position = current_position  # Actualizar posición previa

        # Determinar si el episodio ha terminado
        # done = self.distance_traveled >= self.long  # Finaliza cuando la distancia recorrida alcanza la longitud total

        self.steps += 1
        # Termina el episodio después de max_steps o ha pasado por todos los waypoints (or self.numWaypoints >= len(self.waypoints))
        if(self.steps >= self.max_steps):
            done = True
        else:
            done = False

        if self.episode_count % self.frecTraj == 1:    # 1 de cada X (self.frecTraj)
            self.positions.append((self.model_position[0], self.model_position[1]))
        
        self.send_action(action[0], action[1])
        # time.sleep(0.025)

        # Calculamos la recompensa
        # s_time = time.time()
        reward, truncated = self.reward_func()
        # f_time = time.time()
        # dif = f_time-s_time
        # self.times += dif

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # print("Tiempo actual:", dif)
        
        # if (done or truncated):
        #     print("Tiempo medio", self.times/self.steps)
        
        return self.image, reward, done, truncated, {}

    def send_action(self, steering_angle, throttle):
        self.speed = throttle
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.drive.steering_angle = steering_angle
        ack_msg.drive.speed = throttle
        self.ack_publisher.publish(ack_msg)

    def callback_model_states(self, data):
        """Callback para recibir la posición y orientación del robot desde Gazebo."""
        try:
            robot_index = data.name.index('racecar')
            position = data.pose[robot_index].position
            orientation = data.pose[robot_index].orientation

            # Almacenar posición y orientación en variables internas
            self.model_position = np.array([position.x, position.y, position.z])
            self.model_orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        except ValueError:
            rospy.logerr("El modelo 'racecar' no se encuentra en Gazebo.")

    def get_model_state(self):
        """Devuelve el estado del modelo cuando se le pida"""
        return self.model_position, self.model_orientation

    def callback_image(self, data):
        bridge = CvBridge()
        self.image = cv2.resize(bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'), (160, 120))

    def reward_func(self):
        """
        Calculates reward based on:
        1. Track center adherence
        2. Carrot-on-stick approach using hyperbolic reward
        3. Progressive waypoint completion
        """
        total_reward = 0
        if len(self.waypoints) < 2:
            return -1, True  # Not enough waypoints to calculate direction
        
        speed = self.speed
        if speed < 0:
            print("VELOCIDAD")
            return -1000, True  # Negative speed (reverse) is incorrect behavior


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
        
        # Calcular el coseno del ángulo entre el vector de dirección y el vector de orientación del robot
        cos_angle = np.dot(direction_vector_normalized, car_vector)

        if(cos_angle < 0):
            print("DIRECCIÓN CONTRARIA")
            return -(self.max_steps - self.steps), True

        # Check if out of bounds
        distance_to_center = np.linalg.norm(robot_pos - nearest_waypoint)
        max_distance = self.thickness / 2
        if distance_to_center > max_distance:
            print("FUERA DE PISTA")
            return -(self.max_steps - self.steps), True

        # Track completed waypoints and calculate progressive waypoint reward
        if self.prevWaypoint != nearest_index:
            diff = (nearest_index - self.prevWaypoint) %  len(self.waypoints)
            waypoint_increment = diff
            self.numWaypoints += waypoint_increment
                
            self.prevWaypoint = nearest_index

        # Bonus for completing all waypoints (NO MAS REWARD POR PASAR META, REWARD POR AVANZAR) wp increment por reward
        if self.numWaypoints >= len(self.waypoints):
            total_reward += self.max_steps - self.steps
            print("All waypoints completed, bonus reward:", self.max_steps - self.steps)
            self.numWaypoints = 0

        # Calculate center reward (TANGENCIAL, ANTES HE HECO EL COSENO, PUES AHORA SENO)
        center_reward = 1.0 - (distance_to_center / max_distance)
        
        # Proximity reward (carrot-on-stick approach)
        next_index = (nearest_index + self.distance) % len(self.waypoints)
        next_waypoint = self.waypoints[next_index]
        
        distanceToNext = np.linalg.norm(robot_pos - next_waypoint)
        distanceToNextPrev = np.linalg.norm(self.prevPos - next_waypoint)
        
        self.prevPos = robot_pos

        # Hyperbolic reward function that increases as car gets closer to target
        proximity_reward = distanceToNextPrev - distanceToNext
        # print("Prev:", distanceToNextPrev)
        # print("Actual:", distanceToNext)
        
        # Combine rewards (without direction and speed components)
        total_reward = (
            center_reward * 0.5 +                  # Stay centered on track (increased weight)
            proximity_reward * self.weightProx +     # Chase the carrot (target waypoint)
            cos_angle
        )

        # print("Center reward:", center_reward)
        # print("Prox reward:", proximity_reward)
        
        if speed < 0.3:
           total_reward -= 5.0
        elif speed < 1:
            total_reward -= 1.0 / speed
        else:
            total_reward += 1.0

        # print("Total reward:", total_reward)

        return total_reward, False

    def close(self):
        rospy.signal_shutdown("Cierre del entorno DeepRacer.")
