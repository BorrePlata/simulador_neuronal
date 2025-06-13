from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Importar módulos personalizados
from config import *
from models.neurons import NeuronGroups, set_neuron_params, RS_PARAMS, FS_PARAMS, HH_PARAMS, LTS_PARAMS
from models.synapses import SynapseGroups, set_synapse_params
from models.plasticity import PlasticityMechanisms
from network.connectivity import random_connectivity, small_world_connectivity, scale_free_connectivity
from network.stimulation import StimulusGenerator, apply_stimulus
from analysis.spike_analysis import (calculate_firing_rate, calculate_cv_isi, calculate_pairwise_correlation,
                                     calculate_spike_train_synchrony, calculate_spike_triggered_average)
from analysis.network_metrics import (create_adjacency_matrix, calculate_clustering_coefficient, 
                                      calculate_path_length, calculate_small_worldness, 
                                      calculate_modularity, calculate_rich_club_coefficient)
from visualization.plots import (raster_plot, membrane_potential_plot, connectivity_heatmap, 
                                 network_graph, firing_rate_plot, isi_histogram)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulación de Red Neuronal Avanzada")
    parser.add_argument('--neurons', type=str, default='RS_FS', help='Tipos de neuronas (RS_FS, HH, LTS)')
    parser.add_argument('--connectivity', type=str, default='random', help='Tipo de conectividad (random, small_world, scale_free)')
    parser.add_argument('--plasticity', type=str, default='STDP', help='Tipo de plasticidad (STDP, Oja, BCM)')
    parser.add_argument('--stimulus', type=str, default='constant', help='Tipo de estímulo (constant, sinusoidal, poisson)')
    parser.add_argument('--analysis', type=str, nargs='+', default=['firing_rate', 'cv_isi'], help='Análisis a realizar')
    parser.add_argument('--visualizations', type=str, nargs='+', default=['raster', 'membrane'], help='Visualizaciones a generar')
    return parser.parse_args()

# Funciones de setup
def setup_neurons(neuron_type):
    if neuron_type == 'RS_FS':
        neuron_group_exc = NeuronGroups.create_RS_neurons(N_EXC)
        neuron_group_inh = NeuronGroups.create_FS_neurons(N_INH)
        set_neuron_params(neuron_group_exc, RS_PARAMS)
        set_neuron_params(neuron_group_inh, FS_PARAMS)
    elif neuron_type == 'HH':
        neuron_group_exc = NeuronGroups.create_HH_neurons(N_EXC)
        neuron_group_inh = NeuronGroups.create_HH_neurons(N_INH)
        set_neuron_params(neuron_group_exc, HH_PARAMS)
        set_neuron_params(neuron_group_inh, HH_PARAMS)
    elif neuron_type == 'LTS':
        neuron_group_exc = NeuronGroups.create_RS_neurons(N_EXC)
        neuron_group_inh = NeuronGroups.create_LTS_neurons(N_INH)
        set_neuron_params(neuron_group_exc, RS_PARAMS)
        set_neuron_params(neuron_group_inh, LTS_PARAMS)
    else:
        raise ValueError(f"Tipo de neurona no reconocido: {neuron_type}")
    
    return neuron_group_exc, neuron_group_inh

def setup_connectivity(connectivity_type, neuron_group_exc, neuron_group_inh):
    if connectivity_type == 'random':
        synapses_ee = SynapseGroups.create_synapses(neuron_group_exc, neuron_group_exc, 'exc', CONNECTIVITY_EE)
        synapses_ei = SynapseGroups.create_synapses(neuron_group_exc, neuron_group_inh, 'exc', CONNECTIVITY_EI)
        synapses_ie = SynapseGroups.create_synapses(neuron_group_inh, neuron_group_exc, 'inh', CONNECTIVITY_IE)
        synapses_ii = SynapseGroups.create_synapses(neuron_group_inh, neuron_group_inh, 'inh', CONNECTIVITY_II)
    elif connectivity_type == 'small_world':
        # Implementación del mundo pequeño si es necesario
        pass
    elif connectivity_type == 'scale_free':
        # Implementación de la red libre de escala si es necesario
        pass
    else:
        raise ValueError(f"Tipo de conectividad no reconocido: {connectivity_type}")
    
    return synapses_ee, synapses_ei, synapses_ie, synapses_ii

def setup_plasticity(plasticity_type, synapses_ee):
    if plasticity_type == 'STDP':
        PlasticityMechanisms.add_stdp(synapses_ee, tau_pre=TAU_PRE, tau_post=TAU_POST, Apre=A_PRE, Apost=A_POST, wmax=W_MAX)
    elif plasticity_type == 'Oja':
        PlasticityMechanisms.add_oja_rule(synapses_ee, eta=ETA, alpha=ALPHA)
    elif plasticity_type == 'BCM':
        PlasticityMechanisms.add_bcm_rule(synapses_ee, eta=ETA, tau_theta=TAU_THETA)
    else:
        raise ValueError(f"Tipo de plasticidad no reconocido: {plasticity_type}")

def setup_plasticity(plasticity_type, synapses_ee):
    if plasticity_type == 'STDP':
        PlasticityMechanisms.add_stdp(synapses_ee, tau_pre=TAU_PRE, tau_post=TAU_POST, Apre=A_PRE, Apost=A_POST, wmax=W_MAX)
    elif plasticity_type == 'Oja':
        PlasticityMechanisms.add_oja_rule(synapses_ee, eta=ETA, alpha=ALPHA)
    elif plasticity_type == 'BCM':
        PlasticityMechanisms.add_bcm_rule(synapses_ee, eta=ETA, tau_theta=TAU_THETA)
    else:
        raise ValueError(f"Tipo de plasticidad no reconocido: {plasticity_type}")

def setup_stimulus(stimulus_type, neuron_group_exc):
    if stimulus_type == 'constant':
        stimulus = StimulusGenerator.constant_current(neuron_group_exc, amplitude=200*pA, start_time=500*ms, end_time=1500*ms)
    elif stimulus_type == 'sinusoidal':
        stimulus = StimulusGenerator.sinusoidal_current(neuron_group_exc, amplitude=200*pA, frequency=10*Hz, phase=0, start_time=500*ms, end_time=1500*ms)
    elif stimulus_type == 'poisson':
        stimulus = StimulusGenerator.poisson_spike_train(neuron_group_exc, rate=100*Hz, start_time=500*ms, end_time=1500*ms)
    else:
        raise ValueError(f"Tipo de estímulo no reconocido: {stimulus_type}")
    
    apply_stimulus(neuron_group_exc, stimulus)
    return stimulus

def setup_simulation(args):
    neuron_group_exc, neuron_group_inh = setup_neurons(args.neurons)
    synapses_ee, synapses_ei, synapses_ie, synapses_ii = setup_connectivity(args.connectivity, neuron_group_exc, neuron_group_inh)
    
    # Conectar las sinapsis si es necesario antes de aplicar plasticidad
    synapses_ee.connect(p=CONNECTIVITY_EE)  # Conectar sinapsis excitatorias-excitatorias
    
    set_synapse_params(synapses_ee, 'exc')
    set_synapse_params(synapses_ei, 'exc')
    set_synapse_params(synapses_ie, 'inh')
    set_synapse_params(synapses_ii, 'inh')
    
    setup_plasticity(args.plasticity, synapses_ee)
    stimulus = setup_stimulus(args.stimulus, neuron_group_exc)
    
    spike_monitor_exc = SpikeMonitor(neuron_group_exc)
    spike_monitor_inh = SpikeMonitor(neuron_group_inh)
    state_monitor_exc = StateMonitor(neuron_group_exc, 'v', record=range(N_RECORDED_NEURONS))
    
    net = Network(neuron_group_exc, neuron_group_inh, 
                  synapses_ee, synapses_ei, synapses_ie, synapses_ii,
                  spike_monitor_exc, spike_monitor_inh, state_monitor_exc)
    
    return net, spike_monitor_exc, spike_monitor_inh, state_monitor_exc, stimulus

def run_simulation(net):
    net.run(SIMULATION_TIME)

def analyze_results(spike_monitor_exc, spike_monitor_inh, state_monitor_exc, analysis_types):
    results = {}
    
    if 'firing_rate' in analysis_types:
        firing_rate_exc, _ = calculate_firing_rate(spike_monitor_exc.spike_trains().values(), BIN_SIZE, 0*ms, SIMULATION_TIME)
        firing_rate_inh, _ = calculate_firing_rate(spike_monitor_inh.spike_trains().values(), BIN_SIZE, 0*ms, SIMULATION_TIME)
        results['firing_rate_exc'] = np.mean(firing_rate_exc)
        results['firing_rate_inh'] = np.mean(firing_rate_inh)
        print(f"Tasa de disparo promedio (Excitatorias): {results['firing_rate_exc']:.2f} Hz")
        print(f"Tasa de disparo promedio (Inhibitorias): {results['firing_rate_inh']:.2f} Hz")

    if 'cv_isi' in analysis_types:
        cv_isi = calculate_cv_isi(spike_monitor_exc.spike_trains()[0])
        results['cv_isi'] = cv_isi
        print(f"CV ISI (Neurona excitatoria 0): {cv_isi:.2f}")

    if 'correlation' in analysis_types:
        correlation, _ = calculate_pairwise_correlation(spike_monitor_exc.spike_trains()[0], 
                                                        spike_monitor_exc.spike_trains()[1], 
                                                        BIN_SIZE, 0*ms, SIMULATION_TIME)
        results['max_correlation'] = np.max(correlation)
        print(f"Correlación máxima entre dos neuronas excitatorias: {results['max_correlation']:.2f}")

    if 'synchrony' in analysis_types:
        synchrony = calculate_spike_train_synchrony([train for train in spike_monitor_exc.spike_trains().values()])
        results['synchrony'] = synchrony
        print(f"Índice de sincronía de la red: {synchrony:.2f}")

    if 'network_metrics' in analysis_types:
        adj_matrix = create_adjacency_matrix(spike_monitor_exc.spike_trains().values())
        results['clustering'] = calculate_clustering_coefficient(adj_matrix)
        results['path_length'] = calculate_path_length(adj_matrix)
        results['small_worldness'] = calculate_small_worldness(adj_matrix)
        print(f"Coeficiente de clustering: {results['clustering']:.2f}")
        print(f"Longitud de camino promedio: {results['path_length']:.2f}")
        print(f"Índice de mundo pequeño: {results['small_worldness']:.2f}")

    return results

def visualize_results(spike_monitor_exc, spike_monitor_inh, state_monitor_exc, visualization_types):
    if 'raster' in visualization_types:
        raster_plot([spike_monitor_exc, spike_monitor_inh], [range(N_EXC), range(N_INH)])
        plt.title("Raster Plot")
        plt.show()

    if 'membrane' in visualization_types:
        membrane_potential_plot(state_monitor_exc, range(5))
        plt.title("Potencial de Membrana")
        plt.show()

    if 'firing_rate' in visualization_types:
        firing_rate_plot([spike_monitor_exc, spike_monitor_inh], [range(N_EXC), range(N_INH)])
        plt.title("Tasa de Disparo")
        plt.show()

    if 'isi_histogram' in visualization_types:
        isi_histogram([train for train in spike_monitor_exc.spike_trains().values()])
        plt.title("Histograma de Intervalos entre Espigas")
        plt.show()

    if 'connectivity' in visualization_types:
        adj_matrix = create_adjacency_matrix(spike_monitor_exc.spike_trains().values())
        connectivity_heatmap(adj_matrix)
        plt.title("Mapa de Calor de Conectividad")
        plt.show()

    if 'network_graph' in visualization_types:
        adj_matrix = create_adjacency_matrix(spike_monitor_exc.spike_trains().values())
        network_graph(adj_matrix)
        plt.title("Grafo de la Red Neuronal")
        plt.show()

def main():
    args = parse_arguments()
    
    # Configurar la simulación
    net, spike_monitor_exc, spike_monitor_inh, state_monitor_exc, stimulus = setup_simulation(args)

    # Ejecutar la simulación
    run_simulation(net)

    # Analizar resultados
    results = analyze_results(spike_monitor_exc, spike_monitor_inh, state_monitor_exc, args.analysis)

    # Visualizar resultados
    visualize_results(spike_monitor_exc, spike_monitor_inh, state_monitor_exc, args.visualizations)

if __name__ == "__main__":
    main()