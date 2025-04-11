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

        # Sliders
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(0)
        self.speed_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        self.steering_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.steering_slider.setMinimum(-100)
        self.steering_slider.setMaximum(100)
        self.steering_slider.setValue(0)
        self.steering_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        # Labels
        self.speed_label = QtWidgets.QLabel("Speed: 0.0 m/s")
        self.steering_label = QtWidgets.QLabel("Steering Angle: 0.0 rad")

        # Layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_slider)
        layout.addWidget(self.steering_label)
        layout.addWidget(self.steering_slider)
        self.setLayout(layout)

        # Connections
        self.speed_slider.valueChanged.connect(self.update_values)
        self.steering_slider.valueChanged.connect(self.update_values)

        # Timer to publish regularly
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.publish_cmd)
        self.timer.start(100)  # 10Hz

        self.speed = 0.0
        self.steering = 0.0

    def update_values(self):
        self.speed = self.speed_slider.value() / 20.0  # Map 0-100 -> 0-5
        self.steering = self.steering_slider.value() / 100.0  # Map -100 to 100 -> -1.0 to 1.0
        self.speed_label.setText(f"Speed: {self.speed:.1f} m/s")
        self.steering_label.setText(f"Steering Angle: {self.steering:.2f} rad")

    def publish_cmd(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        msg.drive.steering_angle = self.steering
        self.pub.publish(msg)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = AckermannGUI()
    gui.show()
    sys.exit(app.exec_())
