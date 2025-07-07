#!/usr/bin/env python3
"""
2D Data Viewer GUI

Features:
- Drag and drop files or open via menu
- Displays list of files
- Reads 2D data via numpy.loadtxt
- Renders with matplotlib
- Navigate through files with Previous/Next buttons or arrow keys

Dependencies:
- PyQt5
- matplotlib
- numpy

Install with:
    pip install PyQt5 matplotlib numpy
"""
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        plt.tight_layout()

    def plot(self, data):
        self.ax.clear()
        # Display 2D data as image; adjust origin and aspect as needed
        self.ax.imshow(data, aspect='auto', origin='lower')
        self.draw()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Data Viewer")
        self.setAcceptDrops(True)
        self.files = []
        self.current_index = -1

        # Central widget and layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Split into list and plot
        hlayout = QtWidgets.QHBoxLayout()
        layout.addLayout(hlayout)

        # File list widget
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_list_changed)
        hlayout.addWidget(self.list_widget)

        # Matplotlib canvas
        self.canvas = PlotCanvas(self)
        hlayout.addWidget(self.canvas, 1)

        # Navigation buttons
        nav_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(nav_layout)
        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.next_btn = QtWidgets.QPushButton("Next")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)

        # Menu actions
        open_action = QtWidgets.QAction("&Open Files", self)
        open_action.triggered.connect(self.open_files)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)

    def open_files(self):
        # Allow multi-select of text files
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open data files",
            "",
            "Text Files (*.txt *.dat);;All Files (*)"
        )
        if paths:
            self.add_files(paths)

    def add_files(self, paths):
        # Add new files to the list
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.list_widget.addItem(p)
        # If first load, select the first file
        if self.current_index == -1 and self.files:
            self.list_widget.setCurrentRow(0)

    def on_list_changed(self, idx):
        # Load and plot when selection changes
        if 0 <= idx < len(self.files):
            self.current_index = idx
            try:
                data = np.loadtxt(self.files[idx])
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {e}")
                return
            # Ensure 2D
            if data.ndim == 1:
                data = data[np.newaxis, :]
            self.canvas.plot(data)

    def on_prev(self):
        if self.current_index > 0:
            self.list_widget.setCurrentRow(self.current_index - 1)

    def on_next(self):
        if self.current_index < len(self.files) - 1:
            self.list_widget.setCurrentRow(self.current_index + 1)

    def keyPressEvent(self, event):
        # Navigate with arrow keys
        if event.key() == QtCore.Qt.Key_Left:
            self.on_prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self.on_next()
        else:
            super().keyPressEvent(event)

    # Drag and drop support
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_files(paths)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
