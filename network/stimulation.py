from brian2 import *
import numpy as np

class StimulusGenerator:
    @staticmethod
    def constant_current(target_group, amplitude, start_time, end_time):
        """
        Genera un estímulo de corriente constante.

        Args:
        target_group (NeuronGroup): Grupo de neuronas objetivo
        amplitude (Quantity): Amplitud de la corriente
        start_time (Quantity): Tiempo de inicio del estímulo
        end_time (Quantity): Tiempo de finalización del estímulo

        Returns:
        TimedArray: Estímulo de corriente constante
        """
        duration = end_time - start_time
        time_array = np.linspace(0, duration/ms, int(duration/defaultclock.dt) + 1) * ms
        current_array = np.zeros(len(time_array))
        current_array[1:] = amplitude
        
        return TimedArray(current_array * amp, dt=defaultclock.dt)

    @staticmethod
    def sinusoidal_current(target_group, amplitude, frequency, phase, start_time, end_time):
        """
        Genera un estímulo de corriente sinusoidal.

        Args:
        target_group (NeuronGroup): Grupo de neuronas objetivo
        amplitude (Quantity): Amplitud de la corriente
        frequency (Quantity): Frecuencia de la oscilación
        phase (float): Fase inicial en radianes
        start_time (Quantity): Tiempo de inicio del estímulo
        end_time (Quantity): Tiempo de finalización del estímulo

        Returns:
        TimedArray: Estímulo de corriente sinusoidal
        """
        duration = end_time - start_time
        time_array = np.linspace(0, duration/ms, int(duration/defaultclock.dt) + 1) * ms
        current_array = amplitude * np.sin(2 * np.pi * frequency * time_array + phase)
        
        return TimedArray(current_array * amp, dt=defaultclock.dt)

    @staticmethod
    def poisson_spike_train(target_group, rate, start_time, end_time):
        """
        Genera un tren de espigas Poisson.

        Args:
        target_group (NeuronGroup): Grupo de neuronas objetivo
        rate (Quantity): Tasa de disparo media
        start_time (Quantity): Tiempo de inicio del estímulo
        end_time (Quantity): Tiempo de finalización del estímulo

        Returns:
        SpikeGeneratorGroup: Grupo generador de espigas Poisson
        """
        duration = end_time - start_time
        n_spikes = int(rate * duration)
        spike_times = np.random.uniform(start_time/ms, end_time/ms, n_spikes) * ms
        spike_times = np.sort(spike_times)
        spike_indices = np.random.randint(0, len(target_group), n_spikes)
        
        return SpikeGeneratorGroup(len(target_group), spike_indices, spike_times)

    @staticmethod
    def gaussian_noise_current(target_group, mean, std, start_time, end_time):
        """
        Genera un estímulo de corriente con ruido gaussiano.

        Args:
        target_group (NeuronGroup): Grupo de neuronas objetivo
        mean (Quantity): Media de la corriente
        std (Quantity): Desviación estándar de la corriente
        start_time (Quantity): Tiempo de inicio del estímulo
        end_time (Quantity): Tiempo de finalización del estímulo

        Returns:
        TimedArray: Estímulo de corriente con ruido gaussiano
        """
        duration = end_time - start_time
        time_array = np.linspace(0, duration/ms, int(duration/defaultclock.dt) + 1) * ms
        noise = np.random.normal(mean/amp, std/amp, len(time_array))
        
        return TimedArray(noise * amp, dt=defaultclock.dt)

    @staticmethod
    def ou_noise_current(target_group, mean, std, tau, start_time, end_time):
        """
        Genera un estímulo de corriente con ruido de Ornstein-Uhlenbeck.

        Args:
        target_group (NeuronGroup): Grupo de neuronas objetivo
        mean (Quantity): Media de la corriente
        std (Quantity): Desviación estándar de la corriente
        tau (Quantity): Constante de tiempo del proceso OU
        start_time (Quantity): Tiempo de inicio del estímulo
        end_time (Quantity): Tiempo de finalización del estímulo

        Returns:
        TimedArray: Estímulo de corriente con ruido de Ornstein-Uhlenbeck
        """
        duration = end_time - start_time
        dt = defaultclock.dt
        n_steps = int(duration / dt)
        noise = np.zeros(n_steps)
        noise[0] = np.random.normal(mean/amp, std/amp)
        
        for i in range(1, n_steps):
            dx = (mean/amp - noise[i-1]) * dt/tau + std/amp * np.sqrt(2*dt/tau) * np.random.normal()
            noise[i] = noise[i-1] + dx
        
        return TimedArray(noise * amp, dt=dt)

    @staticmethod
    def naturalistic_input(target_group, input_function, start_time, end_time, **kwargs):
        """
        Genera un estímulo naturalista basado en una función definida por el usuario.

        Args:
        target_group (NeuronGroup): Grupo de neuronas objetivo
        input_function (callable): Función que genera el estímulo
        start_time (Quantity): Tiempo de inicio del estímulo
        end_time (Quantity): Tiempo de finalización del estímulo
        **kwargs: Argumentos adicionales para input_function

        Returns:
        TimedArray: Estímulo naturalista
        """
        duration = end_time - start_time
        time_array = np.linspace(0, duration/ms, int(duration/defaultclock.dt) + 1) * ms
        stimulus = input_function(time_array, **kwargs)
        
        return TimedArray(stimulus * amp, dt=defaultclock.dt)

def apply_stimulus(target_group, stimulus, variable='I'):
    """
    Aplica un estímulo a un grupo de neuronas.

    Args:
    target_group (NeuronGroup): Grupo de neuronas objetivo
    stimulus: Estímulo a aplicar (TimedArray o SpikeGeneratorGroup)
    variable (str): Variable a la que se aplica el estímulo (por defecto 'I' para corriente)

    Returns:
    None
    """
    if isinstance(stimulus, TimedArray):
        # Definir el estímulo como una variable accesible dentro del grupo
        target_group.namespace['stimulus'] = stimulus
        
        # En lugar de añadir una nueva variable, solo asignamos el estímulo si ya existe la variable
        if variable in target_group.variables:
            target_group.run_regularly(f'{variable} = stimulus(t)', when='start')
        else:
            raise ValueError(f"La variable '{variable}' no existe en el grupo de neuronas.")
    elif isinstance(stimulus, SpikeGeneratorGroup):
        synapses = Synapses(stimulus, target_group, on_pre=f'{variable} += 1*nA')
        synapses.connect(j='i')
    else:
        raise ValueError("Tipo de estímulo no soportado")