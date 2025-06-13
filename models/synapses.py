#models/synapses.py
from brian2 import *
from config import *

class SynapseModels:
    @staticmethod
    def simple_synapse():
        """
        Modelo de sinapsis simple con conductancia post-sináptica.
        """
        eqs = '''
        w : siemens    # Peso sináptico
        '''
        on_pre = 'g_post += w'
        return eqs, on_pre

    @staticmethod
    def exc_synapse():
        """
        Modelo de sinapsis excitatoria con dinámica de conductancia AMPA.
        """
        eqs = '''
        dg/dt = -g/tau_AMPA : siemens (clock-driven)
        w : siemens    # Peso sináptico
        tau_AMPA : second
        '''
        on_pre = 'g += w'
        return eqs, on_pre

    @staticmethod
    def inh_synapse():
        """
        Modelo de sinapsis inhibitoria con dinámica de conductancia GABA.
        """
        eqs = '''
        dg/dt = -g/tau_GABA : siemens (clock-driven)
        w : siemens    # Peso sináptico
        tau_GABA : second
        '''
        on_pre = 'g += w'
        return eqs, on_pre

    @staticmethod
    def stdp_synapse():
        """
        Modelo de sinapsis con plasticidad STDP.
        """
        eqs = '''
        w : siemens    # Peso sináptico
        dapre/dt = -apre/tau_pre : 1 (event-driven)
        dapost/dt = -apost/tau_post : 1 (event-driven)
        tau_pre : second
        tau_post : second
        A_pre : 1
        A_post : 1
        w_max : siemens
        '''
        on_pre = '''
        g_post += w
        apre += A_pre
        w = clip(w + apost, 0, w_max)
        '''
        on_post = '''
        apost += A_post
        w = clip(w + apre, 0, w_max)
        '''
        return eqs, on_pre, on_post

    @staticmethod
    def gap_junction():
        """
        Modelo de sinapsis eléctrica (gap junction).
        """
        eqs = '''
        w : siemens    # Conductancia de la gap junction
        Igap = w * (v_post - v_pre) : amp (summed)
        '''
        return eqs

    @staticmethod
    def tsodyks_markram_synapse():
        """
        Modelo de sinapsis de Tsodyks-Markram con depresión y facilitación.
        """
        eqs = '''
        dx/dt = (1 - x)/tau_d : 1 (clock-driven)
        du/dt = (U - u)/tau_f : 1 (clock-driven)
        w : 1    # Peso sináptico base
        U : 1
        tau_d : second
        tau_f : second
        '''
        on_pre = '''
        g_post += w * u * x
        x -= u * x
        u += U * (1 - u)
        '''
        return eqs, on_pre

class SynapseGroups:
    @staticmethod
    def create_synapses(source, target, model, connection_probability, **kwargs):
        """
        Crea un grupo de sinapsis entre poblaciones de neuronas.
        """
        if model == 'simple':
            eqs, on_pre = SynapseModels.simple_synapse()
            S = Synapses(source, target, model=eqs, on_pre=on_pre, **kwargs)
        elif model == 'exc':
            eqs, on_pre = SynapseModels.exc_synapse()
            S = Synapses(source, target, model=eqs, on_pre=on_pre, **kwargs)
        elif model == 'inh':
            eqs, on_pre = SynapseModels.inh_synapse()
            S = Synapses(source, target, model=eqs, on_pre=on_pre, **kwargs)
        elif model == 'stdp':
            eqs, on_pre, on_post = SynapseModels.stdp_synapse()
            S = Synapses(source, target, model=eqs, on_pre=on_pre, on_post=on_post, **kwargs)
        elif model == 'gap':
            eqs = SynapseModels.gap_junction()
            S = Synapses(source, target, model=eqs, **kwargs)
        elif model == 'tsodyks_markram':
            eqs, on_pre = SynapseModels.tsodyks_markram_synapse()
            S = Synapses(source, target, model=eqs, on_pre=on_pre, **kwargs)
        else:
            raise ValueError(f"Modelo de sinapsis desconocido: {model}")
        
        S.connect(p=connection_probability)
        set_synapse_params(S, model)
        return S

# Parámetros sinápticos
SYNAPSE_PARAMS = {
    'simple': {'w': 'rand() * 10*nS'},
    'exc': {'w': 'rand() * 10*nS', 'tau_AMPA': 5*ms},
    'inh': {'w': 'rand() * 10*nS', 'tau_GABA': 10*ms},
    'stdp': {
        'w': 'rand() * 10*nS',
        'tau_pre': 20*ms,
        'tau_post': 20*ms,
        'A_pre': 0.01,
        'A_post': -0.0105,
        'w_max': 20*nS
    },
    'gap': {'w': 'rand() * 5*nS'},
    'tsodyks_markram': {
        'w': 'rand() * 10*nS',
        'U': 0.5,
        'tau_d': 200*ms,
        'tau_f': 600*ms
    }
}

def set_synapse_params(S, model):
    """
    Establece los parámetros para un grupo de sinapsis.
    """
    params = SYNAPSE_PARAMS.get(model, {})
    for param, value in params.items():
        if hasattr(S, param):
            setattr(S, param, value)
        else:
            print(f"Advertencia: El parámetro '{param}' no existe en el grupo de sinapsis.")