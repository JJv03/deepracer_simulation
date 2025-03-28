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
        self.weightProx = 0.7
        self.weightDir = 1
        self.weightSpeed = 0.6

        # Inicializar la posición inicial del robot
        self.initial_position = np.array([-0.5456519086166459, -3.060323716659117, -5.581931699989023e-06])  # x, y, z
        self.initial_orientation = np.array([6.1710519125244906e-06, 2.4181460708256078e-05, -0.2583623974492632, 0.9660480686598593])  # x, y, z, w (cuaternión)
        
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
        Calcula la recompensa basada en:
        1. La proximidad al centro de la pista según el grosor permitido.
        2. Si el robot se sale del recorrido, retorna 0.
        3. La dirección que el robot debe seguir calculada en base a los dos waypoints más cercanos.
        4. Reducir la reward en base a la velocidad
        """
        total_reward = 0
        if len(self.waypoints) < 2:
            return -1, True  # No hay suficientes waypoints para calcular la dirección
        
        speed = self.speed # No debería de ser posible pues el actiónSpace está definido para que como poco sea 0
        if speed < 0:
            return -1000, True  # Velocidad negativa (marcha atrás) es un comportamiento incorrecto

        # Obtener la posición actual del robot
        robot_pos = self.model_position[:2]  # Solo usar las coordenadas x, y

        # Encontrar el waypoint más cercano usando KD-Tree
        _, nearest_index = self.kd_tree.query(robot_pos)
        nearest_waypoint = self.waypoints[nearest_index]

        #print(self.prevWaypoint, nearest_index, self.numWaypoints)
        
        if(self.prevWaypoint != nearest_index):
            diff = (nearest_index - self.prevWaypoint)

            if diff > 0:
                self.numWaypoints += diff % len(self.waypoints)

            # print("Dif waypoints:", nearest_index - self.prevWaypoint)
            # print("Waypoints pasados", self.numWaypoints)
            self.prevWaypoint = nearest_index

        if (self.numWaypoints >= len(self.waypoints)):
            total_reward += self.max_steps - self.steps # Al haber superado todos los waypoints le recompenso con los steps que le faltan aunque siga corriendo
            print("Todos los waypoints han sido superados, recompensa de: ", total_reward)
            self.numWaypoints = 0 # Reiniciar variable a 0

        # Calcular la distancia al centro de la pista
        distance_to_center = np.linalg.norm(robot_pos - nearest_waypoint)
        
        # Calcular la recompensa en base al grosor
        max_distance = self.thickness / 2  # Distancia máxima permitida desde el centro
        if distance_to_center > max_distance:
            return -100, True  # Fuera de la pista

        # Calcular la recompensa de proximidad: 1 en el centro, 0 en el borde
        # proximity_reward = 1 - (distance_to_center / max_distance)
        # print("Reward centro:", proximity_reward)

        # Encontrar el siguiente waypoint más cercano en la secuencia del recorrido
        # next_index = (nearest_index + 1) % len(self.waypoints)  # Siguiente waypoint en el recorrido
        next_index = (nearest_index + self.distance) % len(self.waypoints)  # Siguiente self.distance waypoint en el recorrido
        next_waypoint = self.waypoints[next_index]
        
        distanceToNext = np.linalg.norm(robot_pos - next_waypoint)
                        
        proximity_reward = -(distanceToNext / self.distanceBetweenWaypoints)
        
        # print("Reward centro:", proximity_reward)
        # print("distance:", distanceToNext, "average:", self.distanceBetweenWaypoints)

        next_index = (nearest_index + 1) % len(self.waypoints)  # Siguiente waypoint en el recorrido
        next_waypoint = self.waypoints[next_index]

        # Calcular el vector de dirección (normalizado)
        direction_vector = np.array(next_waypoint) - np.array(nearest_waypoint)
        direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
        # print("Direccion way:", direction_vector_normalized)

        x, y, z, w = self.model_orientation
        # Calcular el ángulo de giro (Yaw) en el plano XY
        theta = np.arctan2(2 * (w * z + x * y), 1 - 2 * (x**2 + z**2))
        
        # Vector dirección en 2D
        car_vector = np.array([np.cos(theta), np.sin(theta)])
        
        # print("Direccion robot:", car_vector)
        
        # Calcular el coseno del ángulo entre el vector de dirección y el vector de orientación del robot
        direction_reward = np.dot(direction_vector_normalized, car_vector) # El coseno del ángulo estará en el rango [-1, 1]
        cos = np.degrees(np.arccos(direction_reward))
        # print("Coseno:", cos)
        # print("Coseno res:", direction_reward)
        
        # speed_reward = np.exp(-abs(speed - 5)) # ActionSpace de hasta 5 de velocidad
        # speed_reward = speed/5
        
        # La recompensa final es una combinación de la proximidad al centro y la alineación con la dirección
        # total_reward = proximity_reward*self.weightProx + direction_reward*self.weightDir
        total_reward = proximity_reward*self.weightProx + direction_reward*self.weightDir

        if speed < 1:
            total_reward -= 1
        elif speed < 2:
            total_reward -= 0.5
        else:
            total_reward += 1
        
        #total_reward = (proximity_reward * direction_reward)*self.weightProxDir + speed_reward*self.weightWaypoints
        #total_reward = (proximity_reward * direction_reward)*self.weightProxDir + waypoints_reward*self.weightWaypoints # * o -, multiplicadores de peso a alguna cosa?
        # print("Reward prox:", proximity_reward)
        # print("Reward dir:", direction_reward)
        # print("Reward:", total_reward)
        
        return total_reward, False

    def close(self):
        rospy.signal_shutdown("Cierre del entorno DeepRacer.")
