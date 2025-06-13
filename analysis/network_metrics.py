# MIT License
# Copyright (c) 2025 Samuel Plata
# This file is part of the Neuronal Interaction Simulator and is licensed under the MIT License.


#analysis/network_metrics.py
import numpy as np
import networkx as nx
from scipy import stats
from sklearn.metrics import mutual_info_score
from brian2.units import *

def create_adjacency_matrix(synapses):
    """
    Crea una matriz de adyacencia a partir de un objeto Synapses de Brian2.

    Args:
    synapses (Synapses): Objeto Synapses de Brian2

    Returns:
    np.array: Matriz de adyacencia
    """
    N = max(max(synapses.i), max(synapses.j)) + 1
    adj_matrix = np.zeros((N, N))
    adj_matrix[synapses.i, synapses.j] = synapses.w
    return adj_matrix

def calculate_degree_distribution(adj_matrix):
    """
    Calcula la distribución de grados de la red.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    tuple: (grados de entrada, grados de salida)
    """
    in_degree = np.sum(adj_matrix, axis=0)
    out_degree = np.sum(adj_matrix, axis=1)
    return in_degree, out_degree

def calculate_clustering_coefficient(adj_matrix):
    """
    Calcula el coeficiente de agrupamiento de la red.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    float: Coeficiente de agrupamiento promedio
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.average_clustering(G)

def calculate_path_length(adj_matrix):
    """
    Calcula la longitud de camino promedio de la red.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    float: Longitud de camino promedio
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.average_shortest_path_length(G)

def calculate_small_worldness(adj_matrix):
    """
    Calcula el índice de mundo pequeño de la red.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    float: Índice de mundo pequeño
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    
    # Generar grafo aleatorio equivalente
    N = adj_matrix.shape[0]
    k = np.mean(np.sum(adj_matrix, axis=1))
    p = k / (N - 1)
    G_rand = nx.erdos_renyi_graph(N, p, directed=True)
    
    C_rand = nx.average_clustering(G_rand)
    L_rand = nx.average_shortest_path_length(G_rand)
    
    return (C / C_rand) / (L / L_rand)

def calculate_betweenness_centrality(adj_matrix):
    """
    Calcula la centralidad de intermediación para cada nodo.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    dict: Centralidad de intermediación para cada nodo
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.betweenness_centrality(G)

def calculate_eigenvector_centrality(adj_matrix):
    """
    Calcula la centralidad de autovector para cada nodo.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    dict: Centralidad de autovector para cada nodo
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.eigenvector_centrality(G)

def calculate_functional_connectivity(spike_trains, bin_size, method='correlation'):
    """
    Calcula la conectividad funcional entre pares de neuronas.

    Args:
    spike_trains (list): Lista de arrays de tiempos de espiga
    bin_size (Quantity): Tamaño del bin temporal
    method (str): Método de cálculo ('correlation' o 'mutual_info')

    Returns:
    np.array: Matriz de conectividad funcional
    """
    N = len(spike_trains)
    max_time = max(max(train) for train in spike_trains if len(train) > 0)
    bins = np.arange(0, max_time/ms + bin_size/ms, bin_size/ms)
    
    binned_trains = [np.histogram(train/ms, bins)[0] for train in spike_trains]
    
    fc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if method == 'correlation':
                fc_matrix[i, j] = stats.pearsonr(binned_trains[i], binned_trains[j])[0]
            elif method == 'mutual_info':
                fc_matrix[i, j] = mutual_info_score(binned_trains[i], binned_trains[j])
            fc_matrix[j, i] = fc_matrix[i, j]
    
    return fc_matrix

def calculate_modularity(adj_matrix):
    """
    Calcula la modularidad de la red y detecta comunidades.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    float: Modularidad de la red
    list: Lista de comunidades detectadas
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    communities = nx.community.greedy_modularity_communities(G)
    modularity = nx.community.modularity(G, communities)
    return modularity, communities

def calculate_rich_club_coefficient(adj_matrix, k):
    """
    Calcula el coeficiente de club rico para un grado k dado.

    Args:
    adj_matrix (np.array): Matriz de adyacencia
    k (int): Grado mínimo para considerar un nodo como parte del club rico

    Returns:
    float: Coeficiente de club rico
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.rich_club_coefficient(G, normalized=False)[k]

def calculate_participation_coefficient(adj_matrix, communities):
    """
    Calcula el coeficiente de participación para cada nodo.

    Args:
    adj_matrix (np.array): Matriz de adyacencia
    communities (list): Lista de comunidades

    Returns:
    dict: Coeficiente de participación para cada nodo
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Crear un diccionario que mapea cada nodo a su comunidad
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
    
    participation = {}
    for node in G.nodes():
        degree = G.degree(node)
        if degree == 0:
            participation[node] = 0
            continue
        
        sum_ratio = 0
        for comm in range(len(communities)):
            edge_in_comm = sum(1 for neighbor in G.neighbors(node) if node_to_community[neighbor] == comm)
            sum_ratio += (edge_in_comm / degree) ** 2
        
        participation[node] = 1 - sum_ratio
    
    return participation

def calculate_global_efficiency(adj_matrix):
    """
    Calcula la eficiencia global de la red.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    float: Eficiencia global de la red
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.global_efficiency(G)

def calculate_local_efficiency(adj_matrix):
    """
    Calcula la eficiencia local para cada nodo de la red.

    Args:
    adj_matrix (np.array): Matriz de adyacencia

    Returns:
    dict: Eficiencia local para cada nodo
    """
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    return nx.local_efficiency(G)