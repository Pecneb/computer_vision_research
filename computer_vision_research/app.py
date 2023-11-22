import logging
import os
import sys

from PyQt5.QtWidgets import (QAction, QApplication, QGridLayout, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow, QMessageBox,
                             QPushButton, QToolBar, QVBoxLayout, QWidget)

logging.basicConfig(level=logging.DEBUG)

class MyApp(QMainWindow):
    """Main application class. Tha implements
    the detection and tracking interface.

    Derives:
        QMainWindow : PyQt5 main window class 
    """
    def __init__(self):
        super(MyApp, self).__init__()
        self.setWindowTitle("Visualizer")
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = MyApp()
    # window.show()

    app.exec()