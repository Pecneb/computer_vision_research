import sys
import os
import logging
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow,
                             QWidget, 
                             QPushButton, 
                             QMessageBox,
                             QGridLayout,
                             QHBoxLayout,
                             QVBoxLayout,
                             QToolBar,
                             QAction,
                             QLineEdit,
                             QLabel)

logging.basicConfig(level=logging.DEBUG)

class MyApp(QMainWindow):
    """Main application class. Tha implements
    the detection and tracking interface.

    Derives:
        QMainWindow : PyQt5 main window class 
    """
    def __init__(self):
        super(MyApp, self).__init__()
        self.initUI()

    def initUI(self):
        ### Main Window ###
        self.setWindowTitle("Detection and Tracking")
        self.setGeometry(300, 300, 1200, 600)

        ### Toolbar ###
        self.initToolbar() 

        self.show()

    def initToolbar(self):
        # Createa toolbar object and add it to the main window
        self.toolbar = QToolBar('Toolbar')
        self.addToolBar(self.toolbar)

        # Toolbar button Detection
        self.toolbar_btn_detect = QAction("Detection", self)
        self.toolbar_btn_detect.setStatusTip("Use detection interface")
        self.toolbar_btn_detect.triggered.connect(self.detectionView)
        self.toolbar.addAction(self.toolbar_btn_detect)

        # Toolbar button prediction 
        self.toolbar_btn_prediction = QAction("Prediction", self)
        self.toolbar_btn_prediction.setStatusTip("Use prediction interface")
        self.toolbar_btn_prediction.triggered.connect(self.predictionView)
        self.toolbar.addAction(self.toolbar_btn_prediction)

    def detectionView(self):
        """Switch to Detection tab.
        """

        ### Main view ### 
        vertical_layout = QVBoxLayout() 
        horizontal_layout = QHBoxLayout()
        grid_layout = QGridLayout()

        # Run button setup
        btn_run = QPushButton('Run', self) 
        # btn.resize(btn2.sizeHint())
        horizontal_layout.addWidget(btn_run)

        # show toggle button
        show_toggle = QPushButton('Show', self)
        show_toggle.setCheckable(True)
        horizontal_layout.addWidget(show_toggle)

        # resume toggle button
        resume_toggle = QPushButton('Resume', self)
        resume_toggle.setCheckable(True)
        horizontal_layout.addWidget(resume_toggle)

        # device toggle button
        device_switch = QPushButton('GPU', self)
        device_switch.setCheckable(True)
        horizontal_layout.addWidget(device_switch)

        # add horizontal layout to vertical layout
        vertical_layout.addLayout(horizontal_layout)

        # Source video input box
        source_input = QLineEdit("Source", self)
        grid_layout.addWidget(source_input, 1, 2)
        source_input_label = QLabel("Source", self)
        grid_layout.addWidget(source_input_label, 1, 1)

        # k_velocity input text
        k_velocity_input = QLineEdit("10", self)
        grid_layout.addWidget(k_velocity_input, 2, 2)
        k_velocity_input_label = QLabel("k-velocity", self)
        grid_layout.addWidget(k_velocity_input_label, 2, 1)

        # k_acceleration input text
        k_acceleration_input = QLineEdit("2", self)
        grid_layout.addWidget(k_acceleration_input, 3, 2)
        k_acceleration_input_label = QLabel("k-acceleration", self)
        grid_layout.addWidget(k_acceleration_input_label, 3, 1)

        # output path input text
        output_path_input = QLineEdit("Output path", self)
        grid_layout.addWidget(output_path_input, 4, 2)
        output_path_input_label = QLabel("Output", self)
        grid_layout.addWidget(output_path_input_label, 4, 1)

        # add grid layout to vertical layout
        vertical_layout.addLayout(grid_layout)

        ### Widget ###
        self.detection_widget = QWidget()
        self.detection_widget.setLayout(vertical_layout)
        self.setCentralWidget(self.detection_widget)

        def run_button_clicked():
            """Run detection on video.
            """
            run_alert = QMessageBox(self)
            run_alert.setWindowTitle("Run")
            run_alert.setText("Detection started")
            run_alert.exec()
            logging.debug(f"Show video: {show_toggle.isChecked()}")
            logging.debug(f"Resume video: {resume_toggle.isChecked()}")
            logging.debug(f"GPU toggle: {device_switch.isChecked()}")
            logging.debug(f"Video source: {source_input.text()}")
            logging.debug(f"Database output path: {output_path_input.text()}")
            logging.debug(f"k-velocity: {k_velocity_input.text()}")
            logging.debug(f"k-acceleration: {k_acceleration_input.text()}")
            os.system(f"python3 detector.py {source_input.text()} --output {output_path_input.text()} --k-velocity {k_velocity_input.text()} --k-acceleration {k_acceleration_input.text()} {'--show' if show_toggle.isChecked() else ''} {'--resume' if resume_toggle.isChecked() else ''} --device {'cuda' if device_switch.isChecked() else 'cpu'}") 

        btn_run.clicked.connect(run_button_clicked)


    def predictionView(self):
        layout = QGridLayout()

        ### Widget ###
        self.prediction_widget = QWidget()
        self.prediction_widget.setLayout(layout)
        self.setCentralWidget(self.prediction_widget)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MyApp()
    # window.show()

    app.exec()