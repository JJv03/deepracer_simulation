import os
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

class DeepRacerEnv(gym.Env):
    def __init__(self, waypoints, thickness):
        super(DeepRacerEnv, self).__init__()
        self.steps = 0
        self.max_steps = 10000
        self.ack_publisher = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, queue_size=100)
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self.callback_image)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_model_states)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.init_node('deepracer_rl', anonymous=True)
        
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.model_position = np.zeros(3)
        self.model_orientation = np.zeros(4)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0]), high=np.array([1.0, 3.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        self.waypoints = waypoints
        self.thickness = thickness

        # Inicializar la posición inicial del robot
        self.initial_position = np.array([-0.5456519086166459, -3.060323716659117, -5.581931699989023e-06])  # x, y, z
        self.initial_orientation = np.array([6.1710519125244906e-06, 2.4181460708256078e-05, -0.2583623974492632, 0.9660480686598593])  # x, y, z, w (cuaternión)


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
        
        self.reward = None
        self.done = False
        self.state = None
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.steps = 0

        # Reiniciar posición en Gazebo
        self.reset_model_state()

        # Inicializa el estado con valores aleatorios en los rangos definidos
        self.send_action(0, 0)  # Establece el acelerador a 0
        print("RESET!")
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

        self.steps += 1
        done = self.steps >= self.max_steps  # Termina el episodio después de max_steps

        self.send_action(action[0], action[1])
        time.sleep(0.1)

        # Calculamos la recompensa
        reward = self.reward_func()
        print(reward)

        # Si el robot se sale del recorrido, reiniciamos
        if reward == 0.0:  # Fuera del recorrido
            self.reset()  # Reiniciar la posición
            truncated = True  # El episodio se truncó porque el robot se salió de la pista
        else:
            truncated = False  # El episodio continúa normalmente

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        return self.image, reward, done, truncated, {}


    def send_action(self, steering_angle, throttle):
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
        """
        if len(self.waypoints) == 0:
            return 0.0

        # Obtener la posición actual del robot
        robot_pos = self.model_position[:2]  # Solo usar las coordenadas x, y

        # Encontrar el waypoint más cercano al robot
        distances_to_waypoints = [np.linalg.norm(robot_pos - np.array(waypoint[:2])) for waypoint in self.waypoints]
        nearest_index = np.argmin(distances_to_waypoints)
        nearest_waypoint = self.waypoints[nearest_index]

        # Calcular la distancia al centro de la pista
        track_center = np.array(nearest_waypoint[:2])  # Centro de la pista en el waypoint más cercano
        distance_to_center = np.linalg.norm(robot_pos - track_center)

        # Calcular la recompensa en base al grosor
        max_distance = self.thickness / 2  # Distancia máxima permitida desde el centro
        if distance_to_center > max_distance:
            return 0.0  # Fuera de la pista

        # Recompensa proporcional: 1 en el centro, 0 en el borde
        reward = 1 - (distance_to_center / max_distance)
        return reward

    def close(self):
        rospy.signal_shutdown("Cierre del entorno DeepRacer.")
