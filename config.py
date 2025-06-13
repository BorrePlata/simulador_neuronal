from brian2 import *

# Parámetros generales de la simulación
SIMULATION_TIME = 2000 * ms
DT = 0.1 * ms  # Paso de tiempo de la simulación

# Parámetros de la red neuronal
N_EXC = 800  # Número de neuronas excitatorias
N_INH = 200  # Número de neuronas inhibitorias

# Parámetros neuronales generales
TAU_M = 10 * ms  # Constante de tiempo de la membrana
EL_EXC = -70 * mV  # Potencial de reposo excitatorio
EL_INH = -70 * mV  # Potencial de reposo inhibitorio
VT_EXC = -50 * mV  # Umbral de excitación excitatorio (ajustado para disparos más fáciles)
VT_INH = -50 * mV  # Umbral de excitación inhibitorio (ajustado para disparos más fáciles)
VR = -75 * mV  # Potencial de reset
REFRACTORY_PERIOD = 1 * ms  # Periodo refractario (ajustado para disparos más frecuentes)

# Parámetros de ruido
SIGMA_EXC = 5 * mV  # Amplitud del ruido excitatorio
SIGMA_INH = 3 * mV  # Amplitud del ruido inhibitorio

# Parámetros sinápticos
ESYN_EXC = 0 * mV  # Potencial sináptico excitatorio
ESYN_INH = -80 * mV  # Potencial sináptico inhibitorio
TAU_SYN_EXC = 5 * ms  # Tiempo sináptico excitatorio
TAU_SYN_INH = 10 * ms  # Tiempo sináptico inhibitorio

# Parámetros de los canales iónicos
G_NA = 100 * nS  # Conductancia máxima del canal de sodio
G_K = 30 * nS  # Conductancia máxima del canal de potasio
E_NA = 55 * mV  # Potencial de equilibrio del sodio
E_K = -90 * mV  # Potencial de equilibrio del potasio

# Parámetros de conectividad
CONNECTIVITY_EE = 0.1  # Probabilidad de conexión excitatoria-excitatoria
CONNECTIVITY_EI = 0.1  # Probabilidad de conexión excitatoria-inhibitoria
CONNECTIVITY_IE = 0.1  # Probabilidad de conexión inhibitoria-excitatoria
CONNECTIVITY_II = 0.1  # Probabilidad de conexión inhibitoria-inhibitoria

# Pesos sinápticos iniciales
W_EE_INIT = '0.5*nS * rand()'
W_EI_INIT = '0.5*nS * rand()'
W_IE_INIT = '2*nS * rand()'
W_II_INIT = '1*nS * rand()'

# Parámetros de plasticidad (STDP)
TAU_PRE = 20 * ms
TAU_POST = 20 * ms
A_PRE = 0.01 * nS
A_POST = -0.0105 * nS
W_MAX = 2 * nS

# Parámetros de estímulo
STIMULUS_AMPLITUDE = 200 * pA  # Amplitud del estímulo aumentada para asegurar actividad
STIM_START_TIME = 500 * ms
STIM_END_TIME = 1500 * ms
STIMULUS_ON = True

# Parámetros de monitoreo
N_RECORDED_NEURONS = 5  # Número de neuronas para registrar el estado completo

# Parámetros de análisis
BIN_SIZE = 1 * ms  # Tamaño del bin para análisis de correlación cruzada
MAX_LAG = 50 * ms  # Máximo lag para análisis de correlación cruzada

# Parámetros adicionales para futuras expansiones
NEURON_TYPES = ['pyramidal', 'interneuron_fs', 'interneuron_lts', 'stellate', 'chandelier']
SYNAPSE_TYPES = ['AMPA', 'NMDA', 'GABA_A', 'GABA_B']
PLASTICITY_TYPES = ['STDP', 'homeostatic', 'short_term_plasticity']

# Configuración de visualización
FIGURE_SIZE = (20, 16)
RASTER_PLOT_MARKER_SIZE = 2
