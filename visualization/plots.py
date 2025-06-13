#visualization/plots.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from brian2.units import *
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

def raster_plot(spike_monitors, neuron_groups, time_range=None, title="Raster Plot"):
    """
    Genera un raster plot para múltiples grupos de neuronas.
    
    Args:
    spike_monitors (list): Lista de SpikeMonitor de Brian2
    neuron_groups (list): Lista de NeuronGroup de Brian2
    time_range (tuple): Rango de tiempo a mostrar (inicio, fin) en ms
    title (str): Título del gráfico
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(spike_monitors)))
    
    offset = 0
    for i, (monitor, group) in enumerate(zip(spike_monitors, neuron_groups)):
        spike_times = monitor.t/ms
        spike_indices = monitor.i
        if time_range:
            mask = (spike_times >= time_range[0]) & (spike_times <= time_range[1])
            spike_times = spike_times[mask]
            spike_indices = spike_indices[mask]
        plt.plot(spike_times, spike_indices + offset, '.', color=colors[i], markersize=2)
        offset += len(group)
    
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Índice de Neurona')
    plt.title(title)
    plt.ylim(0, offset)
    if time_range:
        plt.xlim(time_range)
    plt.tight_layout()

def membrane_potential_plot(state_monitor, neuron_indices, time_range=None, title="Potencial de Membrana"):
    """
    Grafica el potencial de membrana para neuronas seleccionadas.
    
    Args:
    state_monitor (StateMonitor): StateMonitor de Brian2
    neuron_indices (list): Índices de las neuronas a graficar
    time_range (tuple): Rango de tiempo a mostrar (inicio, fin) en ms
    title (str): Título del gráfico
    """
    plt.figure(figsize=(12, 6))
    times = state_monitor.t/ms
    
    if time_range:
        mask = (times >= time_range[0]) & (times <= time_range[1])
        times = times[mask]
    
    for i in neuron_indices:
        v = state_monitor.v[i]/mV
        if time_range:
            v = v[mask]
        plt.plot(times, v, label=f'Neurona {i}')
    
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Potencial de Membrana (mV)')
    plt.title(title)
    plt.legend()
    if time_range:
        plt.xlim(time_range)
    plt.tight_layout()

def isi_histogram(spike_trains, bins=50, title="Histograma de Intervalos entre Espigas"):
    """
    Genera un histograma de intervalos entre espigas (ISI).
    
    Args:
    spike_trains (list): Lista de arrays de tiempos de espiga
    bins (int): Número de bins para el histograma
    title (str): Título del gráfico
    """
    all_isis = []
    for train in spike_trains:
        isis = np.diff(train/ms)
        all_isis.extend(isis)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_isis, bins=bins, edgecolor='black')
    plt.xlabel('Intervalo entre Espigas (ms)')
    plt.ylabel('Frecuencia')
    plt.title(title)
    plt.tight_layout()

def connectivity_heatmap(connectivity_matrix, title="Mapa de Calor de Conectividad"):
    """
    Genera un mapa de calor de la matriz de conectividad.
    
    Args:
    connectivity_matrix (array): Matriz de conectividad
    title (str): Título del gráfico
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(connectivity_matrix, cmap='viridis', cbar_kws={'label': 'Fuerza de Conexión'})
    plt.xlabel('Neurona Destino')
    plt.ylabel('Neurona Fuente')
    plt.title(title)
    plt.tight_layout()

def network_graph(adjacency_matrix, pos=None, node_size=300, node_color='lightblue', title="Grafo de Red Neuronal"):
    """
    Visualiza la red neuronal como un grafo.
    
    Args:
    adjacency_matrix (array): Matriz de adyacencia de la red
    pos (dict): Posiciones de los nodos (opcional)
    node_size (int): Tamaño de los nodos
    node_color (str): Color de los nodos
    title (str): Título del gráfico
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    plt.figure(figsize=(12, 8))
    if pos is None:
        pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=node_size, 
            arrowsize=10, arrows=True)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

def firing_rate_plot(spike_monitors, neuron_groups, bin_size=10*ms, time_range=None, title="Tasa de Disparo"):
    """
    Grafica la tasa de disparo promedio para múltiples grupos de neuronas.
    
    Args:
    spike_monitors (list): Lista de SpikeMonitor de Brian2
    neuron_groups (list): Lista de NeuronGroup de Brian2
    bin_size (Quantity): Tamaño del bin para calcular la tasa de disparo
    time_range (tuple): Rango de tiempo a mostrar (inicio, fin) en ms
    title (str): Título del gráfico
    """
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(spike_monitors)))
    
    for i, (monitor, group) in enumerate(zip(spike_monitors, neuron_groups)):
        spike_times = monitor.t/ms
        if time_range:
            mask = (spike_times >= time_range[0]) & (spike_times <= time_range[1])
            spike_times = spike_times[mask]
        
        bins = np.arange(spike_times[0], spike_times[-1], bin_size/ms)
        counts, _ = np.histogram(spike_times, bins)
        rates = counts / (bin_size/second) / len(group)
        plt.plot(bins[:-1], rates, color=colors[i], label=f'Grupo {i+1}')
    
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Tasa de Disparo (Hz)')
    plt.title(title)
    plt.legend()
    if time_range:
        plt.xlim(time_range)
    plt.tight_layout()

def phase_plane_plot(state_monitor, var1, var2, neuron_index, time_range=None, title="Diagrama de Fase"):
    """
    Genera un diagrama de fase para dos variables de estado de una neurona.
    
    Args:
    state_monitor (StateMonitor): StateMonitor de Brian2
    var1, var2 (str): Nombres de las variables a graficar
    neuron_index (int): Índice de la neurona a graficar
    time_range (tuple): Rango de tiempo a mostrar (inicio, fin) en ms
    title (str): Título del gráfico
    """
    plt.figure(figsize=(8, 8))
    times = state_monitor.t/ms
    v1 = getattr(state_monitor, var1)[neuron_index]
    v2 = getattr(state_monitor, var2)[neuron_index]
    
    if time_range:
        mask = (times >= time_range[0]) & (times <= time_range[1])
        v1 = v1[mask]
        v2 = v2[mask]
    
    plt.plot(v1, v2)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(title)
    plt.tight_layout()

def plot_3d_network(adjacency_matrix, pos=None, title="Red Neuronal 3D"):
    """
    Visualiza la red neuronal en 3D.
    
    Args:
    adjacency_matrix (array): Matriz de adyacencia de la red
    pos (dict): Posiciones de los nodos en 3D (opcional)
    title (str): Título del gráfico
    """
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if pos is None:
        pos = nx.spring_layout(G, dim=3)
    
    # Extraer coordenadas x, y, z
    xs = [pos[k][0] for k in sorted(G.nodes())]
    ys = [pos[k][1] for k in sorted(G.nodes())]
    zs = [pos[k][2] for k in sorted(G.nodes())]
    
    # Dibujar nodos
    ax.scatter(xs, ys, zs, c='r', s=100)
    
    # Dibujar aristas
    for edge in G.edges():
        x = np.array((pos[edge[0]][0], pos[edge[1]][0]))
        y = np.array((pos[edge[0]][1], pos[edge[1]][1]))
        z = np.array((pos[edge[0]][2], pos[edge[1]][2]))
        ax.plot(x, y, z, c='b', alpha=0.5)
    
    ax.set_title(title)
    plt.tight_layout()