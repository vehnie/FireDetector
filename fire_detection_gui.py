import sys
import cv2
import tensorflow as tf
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import serial
import serial.tools.list_ports
import time

class FireDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Detection System")
        self.setGeometry(100, 100, 800, 600)

        # Initialize variables
        self.camera_active = False
        self.temp_active = False
        self.buzzer_active = False
        self.fire_detected = False
        self.current_temperature = "N/A"
        self.cap = None
        self.model = None
        self.model_loaded = False
        self.arduino = None
        self.default_temp = tf.expand_dims((25.0 - 20.0) / (100.0 - 20.0), 0)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Create buttons and temperature display
        button_layout = QHBoxLayout()
        
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        self.camera_button.setEnabled(False)  # Disabled until model loads
        button_layout.addWidget(self.camera_button)

        self.temp_button = QPushButton("Start Temperature")
        self.temp_button.clicked.connect(self.toggle_temperature)
        self.temp_button.setEnabled(False)  # Disabled until model loads
        button_layout.addWidget(self.temp_button)

        self.buzzer_button = QPushButton("Enable Buzzer")
        self.buzzer_button.clicked.connect(self.toggle_buzzer)
        self.buzzer_button.setEnabled(False)  # Disabled until model loads
        button_layout.addWidget(self.buzzer_button)

        # Add temperature display label
        self.temp_label = QLabel("Temperature: N/A")
        self.temp_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        button_layout.addWidget(self.temp_label)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("System Status: Loading model...")
        layout.addWidget(self.status_label)

        # Initialize timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Timer for model loading
        self.load_timer = QTimer()
        self.load_timer.timeout.connect(self.load_model)
        self.load_timer.start(100)  # Start loading model after GUI appears
        
        # Connect to Arduino
        self.connect_arduino()

    def load_model(self):
        try:
            if not self.model_loaded:
                self.model = tf.keras.models.load_model('models/model.keras')
                self.model_loaded = True
                self.status_label.setText("Model loaded successfully")
                self.load_timer.stop()  # Stop the loading timer
                # Enable buttons after model is loaded
                self.camera_button.setEnabled(True)
                self.temp_button.setEnabled(True)
                self.buzzer_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Error loading model: {str(e)}")
            self.load_timer.stop()

    def connect_arduino(self):
        try:
            # Auto-detect Arduino port
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                if "Arduino" in port.description or "CH340" in port.description:
                    self.arduino = serial.Serial(port.device, 9600, timeout=0)
                    self.status_label.setText(f"Connected to Arduino on {port.device}")
                    # Start timer to read temperature
                    self.temp_timer = QTimer()
                    self.temp_timer.timeout.connect(self.read_temperature)
                    self.temp_timer.start(1000)  # Read every second
                    break
            if not hasattr(self, 'arduino') or self.arduino is None:
                self.status_label.setText("No Arduino found. Please check connection.")
        except Exception as e:
            self.status_label.setText(f"Error connecting to Arduino: {str(e)}")
            self.arduino = None

    def read_temperature(self):
        if self.arduino and self.temp_active:
            try:
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode('utf-8').strip()
                    print(f"Received data from Arduino: {line}")  # Debug print
                    try:
                        # Parse temperature value
                        temp = float(line)
                        print(f"Parsed temperature: {temp}")  # Debug print
                        if 0 <= temp <= 100:  # Basic validation
                            self.current_temperature = f"{temp:.1f}°C"
                            self.temp_label.setText(f"Temperature: {self.current_temperature}")
                            print(f"Updated temperature display: {self.current_temperature}")  # Debug print
                            # Update the default temperature for the model
                            self.default_temp = tf.expand_dims((temp - 20.0) / (100.0 - 20.0), 0)
                    except ValueError as ve:
                        print(f"Error parsing temperature value: {ve}")  # Debug print
                        pass
            except Exception as e:
                print(f"Error reading temperature: {str(e)}")  # Debug print

    def toggle_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_active = True
                self.camera_button.setText("Stop Camera")
                self.timer.start(30)  # Update every 30ms
        else:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.camera_active = False
            self.camera_button.setText("Start Camera")
            self.video_label.clear()

    def toggle_temperature(self):
        if self.arduino:
            if not self.temp_active:
                self.arduino.write(b'T')  # Send command to start temperature monitoring
                self.temp_active = True
                self.temp_button.setText("Stop Temperature")
            else:
                self.arduino.write(b't')  # Send command to stop temperature monitoring
                self.temp_active = False
                self.temp_button.setText("Start Temperature")
                self.current_temperature = "N/A"
                self.temp_label.setText(f"Temperature: {self.current_temperature}")

    def toggle_buzzer(self):
        if self.arduino:
            if not self.buzzer_active:
                self.buzzer_active = True
                self.buzzer_button.setText("Stop Buzzer")
                # If fire was previously detected, activate the alarm
                if self.fire_detected:
                    self.arduino.write(b'A')
            else:
                self.arduino.write(b'a')  # Deactivate alarm
                self.fire_detected = False  # Reset fire detection state
                self.buzzer_active = False
                self.buzzer_button.setText("Enable Buzzer")

    def preprocess_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(rgb_frame, (224, 224))
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, 0)
        return img

    def update_frame(self):
        if self.cap and self.camera_active and self.model_loaded:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for prediction
                processed_frame = self.preprocess_frame(frame)
                prediction = self.model.predict(
                    [processed_frame, tf.expand_dims(self.default_temp, 0)], 
                    verbose=0
                )
                confidence = float(prediction[0][0])
                is_fire = confidence >= 0.5

                # Update fire detection state and Arduino control
                if is_fire:
                    self.fire_detected = True
                    if self.buzzer_active and self.arduino:
                        self.arduino.write(b'A')

                # Draw results on frame
                text_color = (0, 0, 255) if is_fire else (0, 255, 0)
                status_text = f"Fire: {is_fire} ({confidence:.2%})"
                if self.fire_detected and self.buzzer_active:
                    status_text += " - ALARM ACTIVE"
                cv2.putText(frame, status_text, 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                
                # Add temperature to frame
                cv2.putText(frame, f"Temp: {self.current_temperature}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Convert frame to Qt format and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.arduino:
            self.arduino.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FireDetectionGUI()
    window.show()
    sys.exit(app.exec_())
