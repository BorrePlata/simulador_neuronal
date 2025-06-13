#analysis/spike_analysis.py
import numpy as np
from scipy import stats, signal
from sklearn.metrics import mutual_info_score
from brian2.units import *
import networkx as nx

def calculate_firing_rate(spike_trains, bin_size, t_start, t_end):
    """
    Calcula la tasa de disparo para un conjunto de trenes de espiga.

    Args:
    spike_trains (list): Lista de arrays de tiempos de espiga
    bin_size (Quantity): Tamaño del bin temporal
    t_start (Quantity): Tiempo de inicio del análisis
    t_end (Quantity): Tiempo de fin del análisis

    Returns:
    array: Tasa de disparo promedio en Hz
    array: Tiempos correspondientes a los bins
    """
    bins = np.arange(t_start/ms, t_end/ms + bin_size/ms, bin_size/ms)
    counts = np.zeros((len(spike_trains), len(bins)-1))
    
    for i, train in enumerate(spike_trains):
        counts[i], _ = np.histogram(train/ms, bins)
    
    firing_rate = counts / (bin_size/second)
    firing_rate_avg = np.mean(firing_rate, axis=0)
    times = bins[:-1] + bin_size/ms/2
    
    return firing_rate_avg, times

def calculate_isi(spike_train):
    """
    Calcula los intervalos entre espigas (ISI) para un tren de espigas.

    Args:
    spike_train (array): Array de tiempos de espiga

    Returns:
    array: Intervalos entre espigas
    """
    return np.diff(spike_train)

def calculate_cv_isi(spike_train):
    """
    Calcula el coeficiente de variación de los intervalos entre espigas.

    Args:
    spike_train (array): Array de tiempos de espiga

    Returns:
    float: Coeficiente de variación del ISI
    """
    isi = calculate_isi(spike_train)
    return np.std(isi) / np.mean(isi)

def calculate_fano_factor(spike_counts):
    """
    Calcula el factor de Fano para un conjunto de conteos de espigas.

    Args:
    spike_counts (array): Array de conteos de espigas

    Returns:
    float: Factor de Fano
    """
    return np.var(spike_counts) / np.mean(spike_counts)

def calculate_pairwise_correlation(spike_train1, spike_train2, bin_size, t_start, t_end):
    """
    Calcula la correlación cruzada entre dos trenes de espigas.

    Args:
    spike_train1, spike_train2 (array): Arrays de tiempos de espiga
    bin_size (Quantity): Tamaño del bin temporal
    t_start (Quantity): Tiempo de inicio del análisis
    t_end (Quantity): Tiempo de fin del análisis

    Returns:
    array: Correlación cruzada
    array: Lags temporales
    """
    bins = np.arange(t_start/ms, t_end/ms + bin_size/ms, bin_size/ms)
    count1, _ = np.histogram(spike_train1/ms, bins)
    count2, _ = np.histogram(spike_train2/ms, bins)
    
    correlation = signal.correlate(count1, count2, mode='full')
    lags = signal.correlation_lags(len(count1), len(count2)) * bin_size/ms
    
    return correlation, lags

def calculate_mutual_information(spike_train1, spike_train2, bin_size, t_start, t_end):
    """
    Calcula la información mutua entre dos trenes de espigas.

    Args:
    spike_train1, spike_train2 (array): Arrays de tiempos de espiga
    bin_size (Quantity): Tamaño del bin temporal
    t_start (Quantity): Tiempo de inicio del análisis
    t_end (Quantity): Tiempo de fin del análisis

    Returns:
    float: Información mutua en bits
    """
    bins = np.arange(t_start/ms, t_end/ms + bin_size/ms, bin_size/ms)
    count1, _ = np.histogram(spike_train1/ms, bins)
    count2, _ = np.histogram(spike_train2/ms, bins)
    
    mi = mutual_info_score(count1, count2)
    return mi

def calculate_spike_triggered_average(stimulus, spike_times, window_size):
    """
    Calcula el promedio disparado por espiga (STA) de un estímulo.

    Args:
    stimulus (array): Array del estímulo
    spike_times (array): Array de tiempos de espiga
    window_size (int): Tamaño de la ventana para el STA

    Returns:
    array: Promedio disparado por espiga
    """
    sta = np.zeros(window_size)
    count = 0
    for spike in spike_times:
        if spike >= window_size and spike < len(stimulus) - window_size:
            sta += stimulus[int(spike-window_size):int(spike)]
            count += 1
    if count > 0:
        sta /= count
    return sta

def calculate_spike_field_coherence(lfp, spike_times, fs, nperseg=256):
    """
    Calcula la coherencia espiga-campo (SFC) entre un LFP y un tren de espigas.

    Args:
    lfp (array): Array del potencial de campo local
    spike_times (array): Array de tiempos de espiga
    fs (float): Frecuencia de muestreo
    nperseg (int): Número de puntos por segmento para el cálculo de la coherencia

    Returns:
    array: Frecuencias
    array: Coherencia espiga-campo
    """
    spike_train = np.zeros_like(lfp)
    spike_indices = (spike_times * fs).astype(int)
    spike_train[spike_indices] = 1
    
    f, Cxy = signal.coherence(lfp, spike_train, fs=fs, nperseg=nperseg)
    return f, Cxy

def calculate_spike_distance(spike_train1, spike_train2):
    """
    Calcula la distancia entre dos trenes de espigas usando la métrica de van Rossum.

    Args:
    spike_train1, spike_train2 (array): Arrays de tiempos de espiga

    Returns:
    float: Distancia entre los trenes de espigas
    """
    # Implementación simplificada de la distancia de van Rossum
    tau = 10 * ms  # Constante de tiempo característica
    
    def exp_kernel(t):
        return np.exp(-t/tau) * (t >= 0)
    
    t = np.linspace(0, max(np.max(spike_train1), np.max(spike_train2)), 1000)
    f1 = np.sum([exp_kernel(t - spike) for spike in spike_train1], axis=0)
    f2 = np.sum([exp_kernel(t - spike) for spike in spike_train2], axis=0)
    
    return np.sqrt(np.sum((f1 - f2)**2) * (t[1] - t[0]))

def calculate_spike_train_synchrony(spike_trains):
    """
    Calcula la sincronía entre múltiples trenes de espigas usando el índice de sincronía de eventos.

    Args:
    spike_trains (list): Lista de arrays de tiempos de espiga

    Returns:
    float: Índice de sincronía
    """
    n_trains = len(spike_trains)
    all_spikes = np.concatenate(spike_trains)
    all_spikes.sort()
    
    coincidences = 0
    window = 1 * ms  # Ventana de coincidencia
    
    for spike in all_spikes:
        count = sum(np.any((train >= spike) & (train < spike + window)) for train in spike_trains)
        if count > 1:
            coincidences += count - 1
    
    rate = np.mean([len(train) for train in spike_trains]) / (all_spikes[-1] - all_spikes[0])
    expected_coincidences = n_trains * (n_trains - 1) * rate * window * len(all_spikes)
    
    return (coincidences - expected_coincidences) / (n_trains * (n_trains - 1) * len(all_spikes))

def calculate_population_vector(spike_counts, preferred_orientations):
    """
    Calcula el vector de población para un conjunto de neuronas con orientaciones preferidas.

    Args:
    spike_counts (array): Conteos de espigas para cada neurona
    preferred_orientations (array): Orientaciones preferidas de cada neurona en radianes

    Returns:
    float: Magnitud del vector de población
    float: Ángulo del vector de población en radianes
    """
    x = np.sum(spike_counts * np.cos(2 * preferred_orientations))
    y = np.sum(spike_counts * np.sin(2 * preferred_orientations))
    
    magnitude = np.sqrt(x**2 + y**2) / np.sum(spike_counts)
    angle = 0.5 * np.arctan2(y, x)
    
    return magnitude, angle

def calculate_spike_train_entropy(spike_train, bin_size):
    """
    Calcula la entropía de un tren de espigas.

    Args:
    spike_train (array): Array de tiempos de espiga
    bin_size (Quantity): Tamaño del bin temporal

    Returns:
    float: Entropía en bits
    """
    bins = np.arange(0, np.max(spike_train) + bin_size, bin_size)
    counts, _ = np.histogram(spike_train, bins)
    prob = counts / np.sum(counts)
    prob = prob[prob > 0]  # Eliminar probabilidades cero
    
    return -np.sum(prob * np.log2(prob))