#network/synapses.py
import numpy as np
from brian2 import *
from scipy.spatial.distance import cdist

def random_connectivity(source, target, p, seed=None):
    """
    Establece conexiones aleatorias entre dos grupos de neuronas.

    Args:
    source (NeuronGroup): Grupo de neuronas fuente
    target (NeuronGroup): Grupo de neuronas objetivo
    p (float): Probabilidad de conexión
    seed (int, optional): Semilla para el generador de números aleatorios

    Returns:
    Synapses: Objeto Synapses de Brian2 con las conexiones establecidas
    """
    if seed is not None:
        if isinstance(seed, float):
            seed = int(seed)
        np.random.seed(seed)
    
    S = Synapses(source, target)
    S.connect(p=p)
    return S

def distance_dependent_connectivity(source, target, max_distance, p_max, spatial_scale, seed=None):
    """
    Establece conexiones basadas en la distancia entre neuronas.

    Args:
    source (NeuronGroup): Grupo de neuronas fuente
    target (NeuronGroup): Grupo de neuronas objetivo
    max_distance (float): Distancia máxima para las conexiones
    p_max (float): Probabilidad máxima de conexión
    spatial_scale (float): Escala espacial de decaimiento de la probabilidad
    seed (int, optional): Semilla para el generador de números aleatorios

    Returns:
    Synapses: Objeto Synapses de Brian2 con las conexiones establecidas
    """
    if seed is not None:
        np.random.seed(seed)
    
    N_source, N_target = len(source), len(target)
    
    # Generar posiciones aleatorias para las neuronas en un espacio 2D
    source_pos = np.random.rand(N_source, 2)
    target_pos = np.random.rand(N_target, 2)
    
    # Calcular distancias entre todas las parejas de neuronas
    distances = cdist(source_pos, target_pos)
    
    # Calcular probabilidades de conexión basadas en la distancia
    connection_probs = p_max * np.exp(-distances / spatial_scale)
    connection_probs[distances > max_distance] = 0
    
    # Establecer conexiones basadas en las probabilidades calculadas
    S = Synapses(source, target)
    S.connect(p=connection_probs)
    return S

def small_world_connectivity(source, target, k, p_rewire, seed=None):
    """
    Establece conexiones siguiendo un modelo de red de mundo pequeño.

    Args:
    source (NeuronGroup): Grupo de neuronas fuente
    target (NeuronGroup): Grupo de neuronas objetivo
    k (int): Número de vecinos más cercanos a conectar inicialmente
    p_rewire (float): Probabilidad de reconexión para cada enlace
    seed (int, optional): Semilla para el generador de números aleatorios

    Returns:
    Synapses: Objeto Synapses de Brian2 con las conexiones establecidas
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = len(source)
    if N != len(target):
        raise ValueError("Los grupos fuente y objetivo deben tener el mismo tamaño para la conectividad de mundo pequeño.")
    
    # Crear conexiones iniciales con los k vecinos más cercanos
    i, j = np.array([(i, j) for i in range(N) for j in range(max(0, i-k//2), min(N, i+k//2+1)) if i != j]).T
    
    # Proceso de reconexión
    for idx in range(len(i)):
        if np.random.rand() < p_rewire:
            j[idx] = np.random.randint(0, N)
    
    S = Synapses(source, target)
    S.connect(i=i, j=j)
    return S

def scale_free_connectivity(source, target, m, alpha, seed=None):
    """
    Establece conexiones siguiendo un modelo de red libre de escala.

    Args:
    source (NeuronGroup): Grupo de neuronas fuente
    target (NeuronGroup): Grupo de neuronas objetivo
    m (int): Número de conexiones a añadir por cada nueva neurona
    alpha (float): Parámetro de preferencia para conexiones de alta conectividad
    seed (int, optional): Semilla para el generador de números aleatorios

    Returns:
    Synapses: Objeto Synapses de Brian2 con las conexiones establecidas
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = len(source)
    if N != len(target):
        raise ValueError("Los grupos fuente y objetivo deben tener el mismo tamaño para la conectividad libre de escala.")
    
    # Inicializar con un pequeño núcleo completamente conectado
    core_size = m + 1
    degrees = np.zeros(N)
    degrees[:core_size] = core_size - 1
    
    # Añadir conexiones para el resto de neuronas
    for i in range(core_size, N):
        # Calcular probabilidades de conexión basadas en los grados actuales
        connection_probs = degrees[:i]**alpha / np.sum(degrees[:i]**alpha)
        
        # Seleccionar m neuronas para conectar
        targets = np.random.choice(i, size=m, replace=False, p=connection_probs)
        degrees[i] += m
        degrees[targets] += 1
    
    # Crear las conexiones
    i, j = [], []
    for source_idx in range(N):
        targets = np.where(np.random.rand(N) < degrees[source_idx] / np.sum(degrees))[0]
        i.extend([source_idx] * len(targets))
        j.extend(targets)
    
    S = Synapses(source, target)
    S.connect(i=i, j=j)
    return S

def structured_connectivity(source, target, connection_pattern, seed=None):
    """
    Establece conexiones estructuradas entre dos grupos de neuronas.

    Args:
    source (NeuronGroup): Grupo de neuronas fuente
    target (NeuronGroup): Grupo de neuronas objetivo
    connection_pattern (ndarray): Matriz de adyacencia que define el patrón de conexiones
    seed (int, optional): Semilla para el generador de números aleatorios

    Returns:
    Synapses: Objeto Synapses de Brian2 con las conexiones establecidas
    """
    if seed is not None:
        np.random.seed(seed)
    
    if connection_pattern.shape != (len(source), len(target)):
        raise ValueError("El patrón de conexión debe tener dimensiones (len(source), len(target))")
    
    i, j = np.where(connection_pattern)
    
    S = Synapses(source, target)
    S.connect(i=i, j=j)
    return S

def layered_connectivity(layers, inter_layer_connections, intra_layer_connections, seed=None):
    """
    Establece conexiones en una red neuronal de múltiples capas.

    Args:
    layers (list of NeuronGroup): Lista de grupos de neuronas representando cada capa
    inter_layer_connections (list of tuple): Lista de tuplas (capa_fuente, capa_objetivo, probabilidad)
    intra_layer_connections (list of tuple): Lista de tuplas (capa, probabilidad)
    seed (int, optional): Semilla para el generador de números aleatorios

    Returns:
    list of Synapses: Lista de objetos Synapses de Brian2 con las conexiones establecidas
    """
    if seed is not None:
        np.random.seed(seed)
    
    connections = []
    
    # Conexiones entre capas
    for source_idx, target_idx, p in inter_layer_connections:
        S = Synapses(layers[source_idx], layers[target_idx])
        S.connect(p=p)
        connections.append(S)
    
    # Conexiones dentro de las capas
    for layer_idx, p in intra_layer_connections:
        S = Synapses(layers[layer_idx], layers[layer_idx])
        S.connect(p=p, condition='i != j')  # Evitar auto-conexiones
        connections.append(S)
    
    return connections