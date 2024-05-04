"""
GUI for biomarker pipeline
@author: Simon Hartmann
"""

from PyQt5 import QtGui, QtWidgets, uic
import sys

class DigBioWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(DigBioWindow, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('DigBio.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.setWindowTitle('DigBio 1.0')  
        self.pathFolder.setText('...')
        self.pathButton.clicked.connect(self.pathButtonClicked)
        self.startButton.clicked.connect(self.startButtonClicked)

###############################################################################
# Button clicked functions
###############################################################################

    def pathButtonClicked(self):

        tmp = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.pathFolder.setText(tmp)

    def startButtonClicked(self):

        print("Not implemented yet")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DigBioWindow()
    app.exec_()