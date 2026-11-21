import sys
import os
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    # Set CUDA environment vars if you have them and want to suppress specific TF/Torch warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
