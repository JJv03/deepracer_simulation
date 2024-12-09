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
    def __init__(self):
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
        1. La proximidad de la línea naranja al centro.
        2. Verifica si la línea naranja está entre dos bordes blancos.
        3. Penaliza si está en zonas verdes.
        """
        if self.image is None:
            return 0.0

        # Convertir la imagen a HSV para detectar colores
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Definir rangos de colores para el naranja (línea central), el verde (zonas a evitar) y blanco (bordes de la pista)
        orange_lower = np.array([0, 50, 50])     # Rango más amplio para el naranja
        orange_upper = np.array([25, 255, 255])
        green_lower = np.array([30, 50, 50])  # Rango verde ajustado para la pista
        green_upper = np.array([100, 255, 255])
        white_lower = np.array([0, 0, 200])  # Definimos un rango de color blanco
        white_upper = np.array([255, 40, 255])

        # Crear máscaras para los colores
        orange_mask = cv2.inRange(hsv_image, orange_lower, orange_upper)
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

        # Mostrar las máscaras para depuración
        cv2.imshow("Máscara Naranja", orange_mask)  # Muestra la máscara para el color naranja
        cv2.imshow("Máscara Verde", green_mask)     # Muestra la máscara para el color verde
        cv2.imshow("Máscara Blanca", white_mask)    # Muestra la máscara para el color blanco
        cv2.waitKey(1)  # Espera una tecla para continuar (necesario para la actualización continua)

        # Calcular el centro de la línea naranja
        orange_moments = cv2.moments(orange_mask)
        if orange_moments['m00'] > 0:
            cx_orange = int(orange_moments['m10'] / orange_moments['m00'])
        else:
            return -1.0  # Penalizar si no se detecta la línea naranja

        print("valor cx_orange: ", cx_orange)

        # Calcular la desviación desde el centro de la imagen (0 es el centro)
        center_offset = abs(cx_orange - (self.image.shape[1] // 2))
        center_reward = 1 - (center_offset / (self.image.shape[1] // 2))  # Normalizado a un rango de 0 a 1

        # Verificar si la línea naranja está dentro de los bordes blancos
        # Supongamos que los bordes blancos están cerca de los lados izquierdo y derecho de la imagen
        white_area_left = np.sum(white_mask[:, :self.image.shape[1] // 3])  # Zona izquierda de los bordes blancos
        white_area_right = np.sum(white_mask[:, 2 * self.image.shape[1] // 3:])  # Zona derecha de los bordes blancos

        if white_area_left == 0 or white_area_right == 0:
            print("Salto por blanco")
            return -1.0  # Penaliza si no está entre los bordes blancos

        # Penalización por estar en áreas verdes
        green_area = np.sum(green_mask) / (self.image.shape[0] * self.image.shape[1] * 255)
        print("valor green_area: ", green_area)
        if green_area > 0.2:  # Si más del 20% de la imagen es verde, penalizamos
            print("Salto por verde")
            return -1.0  # Penalización máxima por estar en la zona verde

        # La recompensa final será la recompensa por centrarse sobre la línea naranja
        reward = center_reward  # 1.0 cuando está centrado, menos cuando está descentrado

        return max(reward, -1.0)  # Asegurar que la recompensa no sea menor que -1.0


    def close(self):
        rospy.signal_shutdown("Cierre del entorno DeepRacer.")
