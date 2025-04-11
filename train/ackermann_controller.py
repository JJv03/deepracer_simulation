#!/usr/bin/env python
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from PyQt5 import QtWidgets, QtCore
import sys

class AckermannGUI(QtWidgets.QWidget):
    def __init__(self):
        super(AckermannGUI, self).__init__()

        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, queue_size=10)
        rospy.init_node('ackermann_controller_gui', anonymous=True)

        self.setWindowTitle("Ackermann Controller")

        # Velocidad y Ángulo iniciales
        self.speed = 0.0
        self.steering = 0.0

        # Etiquetas para mostrar valores actuales
        self.speed_label = QtWidgets.QLabel(f"Speed: {self.speed:.1f} m/s")
        self.steering_label = QtWidgets.QLabel(f"Steering Angle: {self.steering:.2f} rad")

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.speed_label)
        layout.addWidget(self.steering_label)
        self.setLayout(layout)

        # Timer para publicar los valores de velocidad y dirección
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_cmd)
        self.timer.start(100)  # Publicar cada 100 ms (10Hz)

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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = AckermannGUI()
    gui.show()
    sys.exit(app.exec_())
