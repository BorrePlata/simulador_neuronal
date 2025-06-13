from brian2 import *
from config import *

class PlasticityModels:
    @staticmethod
    def stdp_model():
        """
        Modelo de plasticidad dependiente del tiempo de espiga (STDP).
        """
        eqs = '''
        w : 1  # Peso sináptico
        dapre/dt = -apre/tau_pre : 1 (event-driven)
        dapost/dt = -apost/tau_post : 1 (event-driven)
        '''
        pre = '''
        apre += Apre
        w = clip(w + apost, 0, wmax)
        '''
        post = '''
        apost += Apost
        w = clip(w + apre, 0, wmax)
        '''
        return eqs, pre, post

    @staticmethod
    def oja_rule():
        """
        Regla de Oja para plasticidad competitiva.
        """
        eqs = '''
        dw/dt = eta * (pre * post - alpha * post**2 * w) : 1
        '''
        return eqs

    @staticmethod
    def bcm_rule():
        """
        Regla BCM (Bienenstock-Cooper-Munro) para plasticidad dependiente de la tasa.
        """
        eqs = '''
        dw/dt = eta * post * (post - theta) * pre : 1
        dtheta/dt = 1/tau_theta * (post**2 - theta) : 1
        '''
        return eqs

    @staticmethod
    def homeostatic_plasticity():
        """
        Modelo de plasticidad homeostática.
        """
        eqs = '''
        dw/dt = eta * (w_target - w) : 1
        '''
        return eqs

    @staticmethod
    def short_term_plasticity():
        """
        Modelo de plasticidad a corto plazo (depresión y facilitación).
        """
        eqs = '''
        dx/dt = (1 - x)/tau_d : 1 (clock-driven)
        du/dt = (U - u)/tau_f : 1 (clock-driven)
        '''
        pre = '''
        u = U + (1 - U) * u
        r = u * x
        x = clip(x * (1 - u), 0, 1)
        w = w * r
        '''
        return eqs, pre

class PlasticityMechanisms:
    @staticmethod
    def add_stdp(synapses, tau_pre, tau_post, Apre, Apost, wmax):
        """
        Añade plasticidad STDP a un grupo de sinapsis.
        """
        eqs, pre, post = PlasticityModels.stdp_model()
        
        # Definir las ecuaciones en el objeto Synapses
        synapses.equations += eqs
        synapses.pre.code += pre
        synapses.post.code += post
        
        # Definir las constantes de STDP
        synapses.tau_pre = tau_pre
        synapses.tau_post = tau_post
        synapses.Apre = Apre
        synapses.Apost = Apost
        synapses.wmax = wmax

    @staticmethod
    def add_oja_rule(synapses, eta, alpha):
        """
        Añade la regla de Oja a un grupo de sinapsis.
        """
        eqs = PlasticityModels.oja_rule()
        synapses.equations += eqs
        synapses.eta = eta
        synapses.alpha = alpha

    @staticmethod
    def add_bcm_rule(synapses, eta, tau_theta):
        """
        Añade la regla BCM a un grupo de sinapsis.
        """
        eqs = PlasticityModels.bcm_rule()
        synapses.equations += eqs
        synapses.eta = eta
        synapses.tau_theta = tau_theta

    @staticmethod
    def add_homeostatic_plasticity(synapses, eta, w_target):
        """
        Añade plasticidad homeostática a un grupo de sinapsis.
        """
        eqs = PlasticityModels.homeostatic_plasticity()
        synapses.equations += eqs
        synapses.eta = eta
        synapses.w_target = w_target

    @staticmethod
    def add_short_term_plasticity(synapses, U, tau_d, tau_f):
        """
        Añade plasticidad a corto plazo a un grupo de sinapsis.
        """
        eqs, pre = PlasticityModels.short_term_plasticity()
        synapses.equations += eqs
        synapses.pre.code += pre
        synapses.U = U
        synapses.tau_d = tau_d
        synapses.tau_f = tau_f

# Parámetros por defecto para los mecanismos de plasticidad
PLASTICITY_PARAMS = {
    'stdp': {
        'tau_pre': 20*ms,
        'tau_post': 20*ms,
        'Apre': 0.01,
        'Apost': -0.01,
        'wmax': 1
    },
    'oja': {
        'eta': 0.01,
        'alpha': 1
    },
    'bcm': {
        'eta': 0.01,
        'tau_theta': 1000*ms
    },
    'homeostatic': {
        'eta': 0.01,
        'w_target': 0.5
    },
    'stp': {
        'U': 0.5,
        'tau_d': 200*ms,
        'tau_f': 600*ms
    }
}

def set_plasticity_params(synapses, mechanism):
    """
    Establece los parámetros para un mecanismo de plasticidad.
    """
    params = PLASTICITY_PARAMS.get(mechanism, {})
    for param, value in params.items():
        try:
            setattr(synapses, param, value)
        except Exception as e:
            print(f"Error al establecer el parámetro {param}: {e}")
