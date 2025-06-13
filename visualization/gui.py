#gui.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from brian2 import *

# Importar funciones necesarias de otros módulos
from models.neurons import NeuronGroups, set_neuron_params
from models.synapses import SynapseGroups, set_synapse_params
from network.connectivity import random_connectivity
from network.stimulation import StimulusGenerator, apply_stimulus
from analysis.spike_analysis import calculate_firing_rate
from visualization.plots import raster_plot, membrane_potential_plot

class NeuronSimulationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulación de Red Neuronal")
        self.setGeometry(100, 100, 1200, 800)

        # Configuración inicial de la simulación
        self.setup_simulation()

        # Configurar la interfaz gráfica
        self.setup_ui()

        # Timer para actualizar la simulación
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)

    def setup_simulation(self):
        # Configuración inicial de la simulación
        self.N_exc = 800
        self.N_inh = 200
        self.sim_time = 1000*ms
        self.dt = 0.1*ms

        # Crear grupos de neuronas
        self.neuron_group_exc = NeuronGroups.create_RS_neurons(self.N_exc)
        self.neuron_group_inh = NeuronGroups.create_FS_neurons(self.N_inh)

        # Crear conexiones
        self.synapses_ee = random_connectivity(self.neuron_group_exc, self.neuron_group_exc, 0.1)
        self.synapses_ei = random_connectivity(self.neuron_group_exc, self.neuron_group_inh, 0.1)
        self.synapses_ie = random_connectivity(self.neuron_group_inh, self.neuron_group_exc, 0.1)
        self.synapses_ii = random_connectivity(self.neuron_group_inh, self.neuron_group_inh, 0.1)

        # Configurar monitores
        self.spike_monitor_exc = SpikeMonitor(self.neuron_group_exc)
        self.spike_monitor_inh = SpikeMonitor(self.neuron_group_inh)
        self.state_monitor_exc = StateMonitor(self.neuron_group_exc, 'v', record=range(5))

        # Configurar estímulo inicial
        self.stimulus = StimulusGenerator.constant_current(self.neuron_group_exc, 100*pA, 0*ms, self.sim_time)
        apply_stimulus(self.neuron_group_exc, self.stimulus)

        # Crear red y compilar
        self.net = Network(self.neuron_group_exc, self.neuron_group_inh, 
                           self.synapses_ee, self.synapses_ei, self.synapses_ie, self.synapses_ii,
                           self.spike_monitor_exc, self.spike_monitor_inh, self.state_monitor_exc)
        self.net.store()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Controles
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Iniciar Simulación")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Detener Simulación")
        self.stop_button.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stop_button)

        self.stimulus_slider = QSlider(Qt.Horizontal)
        self.stimulus_slider.setRange(0, 200)
        self.stimulus_slider.setValue(100)
        self.stimulus_slider.valueChanged.connect(self.update_stimulus)
        control_layout.addWidget(QLabel("Estímulo (pA):"))
        control_layout.addWidget(self.stimulus_slider)

        self.plot_type = QComboBox()
        self.plot_type.addItems(["Raster Plot", "Potencial de Membrana"])
        self.plot_type.currentIndexChanged.connect(self.update_plot)
        control_layout.addWidget(self.plot_type)

        layout.addLayout(control_layout)

        # Área de gráfico
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def start_simulation(self):
        self.net.restore()
        self.timer.start(100)  # Actualizar cada 100 ms

    def stop_simulation(self):
        self.timer.stop()

    def update_stimulus(self):
        amplitude = self.stimulus_slider.value() * pA
        self.stimulus = StimulusGenerator.constant_current(self.neuron_group_exc, amplitude, 0*ms, self.sim_time)
        apply_stimulus(self.neuron_group_exc, self.stimulus)

    def update_simulation(self):
        self.net.run(10*ms)
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.plot_type.currentText() == "Raster Plot":
            raster_plot([self.spike_monitor_exc, self.spike_monitor_inh], 
                        [self.neuron_group_exc, self.neuron_group_inh], 
                        ax=ax)
        else:
            membrane_potential_plot(self.state_monitor_exc, range(5), ax=ax)

        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = NeuronSimulationGUI()
    gui.show()
    sys.exit(app.exec_())