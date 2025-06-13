#utils/helpers.py
import numpy as np
from brian2 import *
from scipy import signal

def nernst_potential(z, c_in, c_out, temperature=37):
    """
    Calcula el potencial de Nernst para un ion.
    
    Args:
    z (int): Valencia del ion
    c_in (float): Concentración intracelular (en mM)
    c_out (float): Concentración extracelular (en mM)
    temperature (float): Temperatura en grados Celsius (por defecto 37°C)
    
    Returns:
    float: Potencial de Nernst en mV
    """
    R = 8.314 # J/(mol·K)
    F = 96485 # C/mol
    T = temperature + 273.15 # K
    return (R * T / (z * F)) * np.log(c_out / c_in) * 1000 # mV

def goldman_hodgkin_katz(p_na, p_k, p_cl, na_in, na_out, k_in, k_out, cl_in, cl_out, temperature=37):
    """
    Calcula el potencial de membrana usando la ecuación de Goldman-Hodgkin-Katz.
    
    Args:
    p_na, p_k, p_cl (float): Permeabilidades relativas de Na+, K+ y Cl-
    na_in, na_out, k_in, k_out, cl_in, cl_out (float): Concentraciones iónicas intra y extracelulares (en mM)
    temperature (float): Temperatura en grados Celsius (por defecto 37°C)
    
    Returns:
    float: Potencial de membrana en mV
    """
    R = 8.314 # J/(mol·K)
    F = 96485 # C/mol
    T = temperature + 273.15 # K
    
    numerator = p_k * k_out + p_na * na_out + p_cl * cl_in
    denominator = p_k * k_in + p_na * na_in + p_cl * cl_out
    
    return (R * T / F) * np.log(numerator / denominator) * 1000 # mV

def alpha_n(v):
    """Función de activación alpha para la conductancia de potasio."""
    return 0.01 * (v + 55) / (1 - exp(-(v + 55) / 10))

def beta_n(v):
    """Función de inactivación beta para la conductancia de potasio."""
    return 0.125 * exp(-(v + 65) / 80)

def alpha_m(v):
    """Función de activación alpha para la conductancia de sodio."""
    return 0.1 * (v + 40) / (1 - exp(-(v + 40) / 10))

def beta_m(v):
    """Función de inactivación beta para la conductancia de sodio."""
    return 4 * exp(-(v + 65) / 18)

def alpha_h(v):
    """Función de activación alpha para la inactivación del sodio."""
    return 0.07 * exp(-(v + 65) / 20)

def beta_h(v):
    """Función de inactivación beta para la inactivación del sodio."""
    return 1 / (1 + exp(-(v + 35) / 10))

def calculate_firing_rate(spike_trains, bin_size, t_start, t_end):
    """
    Calcula la tasa de disparo para un conjunto de trenes de espiga.
    
    Args:
    spike_trains (list): Lista de arrays de tiempos de espiga
    bin_size (float): Tamaño del bin en segundos
    t_start (float): Tiempo de inicio en segundos
    t_end (float): Tiempo de fin en segundos
    
    Returns:
    array: Tasa de disparo promedio en Hz
    array: Tiempos correspondientes a los bins
    """
    bins = np.arange(t_start, t_end, bin_size)
    counts = np.zeros_like(bins)
    for train in spike_trains:
        counts += np.histogram(train, bins)[0]
    
    firing_rate = counts / (bin_size * len(spike_trains))
    times = bins[:-1] + bin_size/2
    
    return firing_rate, times

def calculate_cv_isi(spike_train):
    """
    Calcula el coeficiente de variación de los intervalos entre espigas.
    
    Args:
    spike_train (array): Array de tiempos de espiga
    
    Returns:
    float: Coeficiente de variación
    """
    isi = np.diff(spike_train)
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

def calculate_coherence(signal1, signal2, fs, nperseg=256):
    """
    Calcula la coherencia entre dos señales.
    
    Args:
    signal1, signal2 (array): Señales de entrada
    fs (float): Frecuencia de muestreo
    nperseg (int): Longitud de cada segmento
    
    Returns:
    array: Frecuencias
    array: Coherencia
    """
    f, Cxy = signal.coherence(signal1, signal2, fs, nperseg=nperseg)
    return f, Cxy

def calculate_mutual_information(X, Y, bins=10):
    """
    Calcula la información mutua entre dos variables.
    
    Args:
    X, Y (array): Variables de entrada
    bins (int): Número de bins para discretización
    
    Returns:
    float: Información mutua en bits
    """
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]
    
    H_X = -np.sum(c_X[c_X>0] * np.log2(c_X[c_X>0] / np.sum(c_X)))
    H_Y = -np.sum(c_Y[c_Y>0] * np.log2(c_Y[c_Y>0] / np.sum(c_Y)))
    H_XY = -np.sum(c_XY[c_XY>0] * np.log2(c_XY[c_XY>0] / np.sum(c_XY)))
    
    return H_X + H_Y - H_XY

def calculate_transfer_entropy(X, Y, delay, k=1, l=1, bins=10):
    """
    Calcula la entropía de transferencia de X a Y.
    
    Args:
    X, Y (array): Series temporales de entrada
    delay (int): Retraso temporal
    k, l (int): Órdenes de Markov para X e Y
    bins (int): Número de bins para discretización
    
    Returns:
    float: Entropía de transferencia en bits
    """
    X = X[:-delay]
    Y = Y[delay:]
    
    def get_state(x, k):
        return ''.join(map(str, np.digitize(x, np.linspace(min(x), max(x), bins))))[-k:]
    
    states_X = [get_state(X[i:i+k], k) for i in range(len(X)-k+1)]
    states_Y = [get_state(Y[i:i+l], l) for i in range(len(Y)-l+1)]
    
    T = len(states_Y)
    
    p_yt = {}
    p_yt_yt = {}
    p_yt_xt_yt = {}
    
    for t in range(T):
        yt = states_Y[t]
        yt1 = states_Y[t-1] if t > 0 else None
        xt = states_X[t]
        
        if yt not in p_yt:
            p_yt[yt] = 0
        p_yt[yt] += 1
        
        if yt1 is not None:
            if (yt, yt1) not in p_yt_yt:
                p_yt_yt[(yt, yt1)] = 0
            p_yt_yt[(yt, yt1)] += 1
            
            if (yt, xt, yt1) not in p_yt_xt_yt:
                p_yt_xt_yt[(yt, xt, yt1)] = 0
            p_yt_xt_yt[(yt, xt, yt1)] += 1
    
    TE = 0
    for yt, xt, yt1 in p_yt_xt_yt:
        p1 = p_yt_xt_yt[(yt, xt, yt1)] / T
        p2 = p_yt_yt[(yt, yt1)] / T
        p3 = p_yt[yt] / T
        
        TE += p1 * np.log2(p1 * p3 / (p2 * p_yt[yt1]))
    
    return TE