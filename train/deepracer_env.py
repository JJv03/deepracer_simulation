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

class DeepRacerEnv(gym.Env):
    def __init__(self, waypoints, thickness):
        super(DeepRacerEnv, self).__init__()
        self.steps = 0
        self.max_steps = 100
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

    def reset(self, seed=None):
        super().reset(seed=seed)
        """
        Reinicia el entorno y devuelve un estado inicial.
        """
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")
        
        self.reward = None
        self.done = False
        self.state = None
        self.image = np.zeros((120, 160, 3), dtype=np.uint8)
        self.steps = 0
        # Inicializa el estado con valores aleatorios en los rangos definidos
        self.send_action(0, 0)  # set the throttle to 0
        return self.image, {}  # Devuelve el estado inicial y un diccionario vacío

    def step(self, action):
        """
        Aplica una acción y avanza un paso en el entorno.
        """

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            #resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.steps += 1
        done = self.steps >= self.max_steps  # Termina el episodio después de max_steps
        
        self.send_action(action[0], action[1])
        time.sleep(0.1)

        # Calcular la recompensa
        reward = self.reward_func()
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        truncated = False

        # Devuelve el nuevo estado, la recompensa, si terminó, y un diccionario vacío
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
        1. La proximidad al waypoint más cercano.
        """
        if len(self.waypoints) == 0:
            return 0.0

        # Obtener la posición actual del robot
        robot_pos = self.model_position[:2]  # Solo usar las coordenadas x, y

        # Calcular la distancia a todos los waypoints
        distances = [np.linalg.norm(robot_pos - np.array(waypoint[:2])) for waypoint in self.waypoints]

        # Encontrar el waypoint más cercano
        min_distance = min(distances)

        # Recompensa: cuanto más cerca esté el robot del waypoint, mejor
        reward = max(1 - min_distance, 0)  # Recompensa positiva basada en la proximidad

        return reward

    def close(self):
        rospy.signal_shutdown("Cierre del entorno DeepRacer.")