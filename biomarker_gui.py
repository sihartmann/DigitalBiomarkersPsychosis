"""
GUI for biomarker pipeline
@author: Simon Hartmann
"""

import sys
import os
from argparse import Namespace
import multiprocessing
from PyQt5 import QtWidgets, uic
from biomarker_pipe import pipeParser, pipe, process_func, make_summary
class DigBioWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(DigBioWindow, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('DigBio.ui', self) # Load the .ui file
        self.show() # Show the GUI
        self.setWindowTitle('DigBio 1.0')

        # Set default values
        self.pathFolder.setText(os.path.expanduser("~"))
        self.modeBox.setCurrentText("all")
        self.verbosityBox.setCurrentText("3")
        self.whisperBox.setCurrentText("base")
        self.nocutBox.setChecked(True)

        # Set GUI signals
        self.pathButton.clicked.connect(self.pathButtonClicked)
        self.startButton.clicked.connect(self.startButtonClicked)

###############################################################################
# Button clicked functions
###############################################################################

    def pathButtonClicked(self):
        self.pathFolder.setText(str(QtWidgets.QFileDialog.getExistingDirectory(
            self,"Select Directory",self.pathFolder.text(),
            QtWidgets.QFileDialog.ShowDirsOnly)))

    def startButtonClicked(self):
        all_args = Namespace(interviews=self.pathFolder.text(), 
                             mode=self.modeBox.currentText(),
                             verbosity=self.verbosityBox.currentText(),
                             overwrite=self.overwriteBox.isChecked(),
                             no_cut=self.nocutBox.isChecked(),
                             whisper_model=self.whisperBox.currentText())

        pipeParserInt = pipeParser()
        num_procs, parsed_all_args = pipeParserInt.parse_args(all_args)
        my_pipes = [pipe() for _ in range(num_procs)]
        summary_queue = multiprocessing.Queue()
        processes = []
        summary_process = multiprocessing.Process(target=make_summary, 
                                                  args=(summary_queue,))
        summary_process.start()
        for i in range(num_procs):
            process = multiprocessing.Process(target=process_func,
                                              args=(summary_queue, my_pipes[i], 
                                              parsed_all_args[i]))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()
        summary_queue.put(None)
        summary_process.join()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = DigBioWindow()
    app.exec_()
