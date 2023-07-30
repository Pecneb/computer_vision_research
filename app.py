import sys
from PyQt5.QtWidgets import (QApplication, 
                             QMainWindow,
                             QWidget, 
                             QPushButton, 
                             QMessageBox,
                             QGridLayout,
                             QToolBar,
                             QAction,
                             QLineEdit,
                             QLabel)

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
        layout = QGridLayout()

        # Run button setup
        btn_run = QPushButton('Run', self) 
        # btn.resize(btn2.sizeHint())
        btn_run.clicked.connect(self.run_button_clicked)
        layout.addWidget(btn_run, 0, 1)

        # Source video input box
        source_input = QLineEdit("Source", self)
        layout.addWidget(source_input, 1, 2)
        source_input_label = QLabel("Source", self)
        layout.addWidget(source_input_label, 1, 1)

        # k_velocity input text
        k_velocity_input = QLineEdit("10", self)
        layout.addWidget(k_velocity_input, 2, 2)
        k_velocity_input_label = QLabel("k-velocity", self)
        layout.addWidget(k_velocity_input_label, 2, 1)

        # k_acceleration input text
        k_acceleration_input = QLineEdit("2", self)
        layout.addWidget(k_acceleration_input, 3, 2)
        k_acceleration_input_label = QLabel("k-acceleration", self)
        layout.addWidget(k_acceleration_input_label, 3, 1)

        ### Widget ###
        self.detection_widget = QWidget()
        self.detection_widget.setLayout(layout)
        self.setCentralWidget(self.detection_widget)

    def run_button_clicked(self):
        run_alert = QMessageBox(self)
        run_alert.setWindowTitle("Run")
        run_alert.setText("Detection started")
        run_alert.exec()

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