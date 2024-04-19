import numpy as np
import matplotlib.pyplot as plt
from PhenomA import *
from constants import *


# function to change mass coordinates
def mass_transform(component_masses):
    m1, m2 = component_masses
    M = m1 + m2
    eta = m1 * m2 / M**2
    return np.array([M, eta])


# modify PhenomA methods so they take component masses as input
# get amplitude of signal in frequency-domain
def get_amp(fs, comp_masses):
    M, eta = mass_transform(comp_masses)
    return Aeff(fs, M, eta, Dl=DL)

# get phase of signal in frequency-domain
def get_phase(fs, comp_masses):
    M, eta = mass_transform(comp_masses)
    return Psieff(fs, M, eta, t0=t0, phi0=phi0)

# get derivative of phase with respect to frequency
def get_phase_deriv(fs, comp_masses):
    M, eta = mass_transform(comp_masses)
    return dPsieff_df(fs, M, eta, t0=t0)


# get frequency-domain waveform
def get_waveform_freq(fs, comp_masses):
    return get_amp(fs, comp_masses) * np.exp(1.j * get_phase(fs, comp_masses))


# plot waveform at measured parameters
fs = np.linspace(20, 1000, 1000)
amps = get_amp(fs, ms_measured)
print(amps)
plt.loglog(fs, amps)
plt.show()

