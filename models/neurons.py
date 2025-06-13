#models/neurons.py
from brian2 import *
from config import *

class NeuronModels:
    @staticmethod
    def HH_base_model():
        """
        Modelo base de Hodgkin-Huxley.
        """
        eqs = '''
        dv/dt = (I - g_Na * m**3 * h * (v - E_Na) - g_K * n**4 * (v - E_K) - g_L * (v - E_L)) / C_m : volt
        dm/dt = alpha_m * (1 - m) - beta_m * m : 1
        dh/dt = alpha_h * (1 - h) - beta_h * h : 1
        dn/dt = alpha_n * (1 - n) - beta_n * n : 1
        alpha_m = 0.1 * (v + 40*mV) / (1 - exp(-(v + 40*mV) / (10*mV))) / ms : Hz
        beta_m = 4 * exp(-(v + 65*mV) / (18*mV)) / ms : Hz
        alpha_h = 0.07 * exp(-(v + 65*mV) / (20*mV)) / ms : Hz
        beta_h = 1 / (1 + exp(-(v + 35*mV) / (10*mV))) / ms : Hz
        alpha_n = 0.01 * (v + 55*mV) / (1 - exp(-(v + 55*mV) / (10*mV))) / ms : Hz
        beta_n = 0.125 * exp(-(v + 65*mV) / (80*mV)) / ms : Hz
        I : amp
        g_Na : siemens
        g_K : siemens
        g_L : siemens
        E_Na : volt
        E_K : volt
        E_L : volt
        C_m : farad
        '''
        return eqs

    @staticmethod
    def RS_model():
        """
        Modelo de neurona de disparo regular (Regular Spiking).
        Basado en el modelo de Izhikevich.
        """
        eqs = '''
        dv/dt = (k * (v - v_r) * (v - v_t) - u + I) / C_m : volt
        du/dt = a * (b * (v - v_r) - u) : amp
        I : amp
        C_m : farad
        v_r : volt
        v_t : volt
        k : siemens/volt
        a : 1/second
        b : siemens
        '''
        return eqs

    @staticmethod
    def FS_model():
        """
        Modelo de neurona de disparo rápido (Fast Spiking).
        Basado en el modelo de Izhikevich.
        """
        return NeuronModels.RS_model()  # Usa las mismas ecuaciones que RS

    @staticmethod
    def LTS_model():
        """
        Modelo de neurona de bajo umbral de disparo (Low Threshold Spiking).
        Basado en el modelo de Izhikevich con corriente de calcio de bajo umbral.
        """
        eqs = '''
        dv/dt = (k * (v - v_r) * (v - v_t) - u + I_Ca + I) / C_m : volt
        du/dt = a * (b * (v - v_r) - u) : amp
        dI_Ca/dt = -I_Ca / tau_Ca : amp
        I : amp
        C_m : farad
        v_r : volt
        v_t : volt
        k : siemens/volt
        a : 1/second
        b : siemens
        tau_Ca : second
        '''
        return eqs

class NeuronGroups:
    @staticmethod
    def create_HH_neurons(N):
        """
        Crea un grupo de neuronas de Hodgkin-Huxley.
        """
        eqs = NeuronModels.HH_base_model()
        G = NeuronGroup(N, eqs, threshold='v > -40*mV', reset='v = -65*mV', method='exponential_euler')
        G.v = '-65*mV'
        G.m = 'alpha_m / (alpha_m + beta_m)'
        G.h = 'alpha_h / (alpha_h + beta_h)'
        G.n = 'alpha_n / (alpha_n + beta_n)'
        return G

    @staticmethod
    def create_RS_neurons(N):
        """
        Crea un grupo de neuronas de disparo regular.
        """
        eqs = NeuronModels.RS_model()
        G = NeuronGroup(N, eqs, threshold='v > 30*mV', reset='v = -65*mV; u += 8*pA', method='rk4')
        G.v = '-65*mV'
        G.u = '0*pA'
        return G

    @staticmethod
    def create_FS_neurons(N):
        """
        Crea un grupo de neuronas de disparo rápido.
        """
        eqs = NeuronModels.FS_model()
        G = NeuronGroup(N, eqs, threshold='v > 30*mV', reset='v = -65*mV; u += 2*pA', method='rk4')
        G.v = '-65*mV'
        G.u = '0*pA'
        return G

    @staticmethod
    def create_LTS_neurons(N):
        """
        Crea un grupo de neuronas de bajo umbral de disparo.
        """
        eqs = NeuronModels.LTS_model()
        G = NeuronGroup(N, eqs, threshold='v > 30*mV', reset='v = -65*mV; u += 8*pA; I_Ca += 5*pA', method='rk4')
        G.v = '-65*mV'
        G.u = '0*pA'
        G.I_Ca = '0*pA'
        return G

# Parámetros específicos para cada tipo de neurona
HH_PARAMS = {
    'g_Na': G_NA,
    'g_K': G_K,
    'g_L': 0.3*nS,
    'E_Na': E_NA,
    'E_K': E_K,
    'E_L': EL_EXC,
    'C_m': 1*uF/cm**2
}

RS_PARAMS = {
    'C_m': 100*pF,
    'v_r': VR,
    'v_t': VT_EXC,
    'k': 0.7*nS/mV,
    'a': 0.03/ms,
    'b': -2*nS
}

FS_PARAMS = {
    'C_m': 20*pF,
    'v_r': VR,
    'v_t': VT_INH,
    'k': 1*nS/mV,
    'a': 0.15/ms,
    'b': 8*nS
}

LTS_PARAMS = {
    'C_m': 100*pF,
    'v_r': VR,
    'v_t': VT_INH,
    'k': 1*nS/mV,
    'a': 0.03/ms,
    'b': 8*nS,
    'tau_Ca': 5*ms
}

def set_neuron_params(G, params):
    """
    Establece los parámetros para un grupo de neuronas.
    """
    for param, value in params.items():
        if hasattr(G, param):
            setattr(G, param, value)
        else:
            print(f"Advertencia: El parámetro '{param}' no existe en el grupo de neuronas.")