import numpy as np
import matplotlib.pyplot as plt
from PhenomA import *
from constants import *


# function to change mass coordinates
def mass_transform(comp_masses):
    m1, m2 = comp_masses
    M = m1 + m2
    eta = m1 * m2 / M**2
    return np.array([M, eta])


# modify PhenomA methods so they take component masses as input
# get amplitude of signal in frequency-domain
def get_amp(fs, params):
    m1, m2, t0, phi0, Dl = params
    M, eta = mass_transform([m1, m2])
    return Aeff(fs, M, eta, Dl=Dl)

# get phase of signal in frequency-domain
def get_phase(fs, params):
    m1, m2, t0, phi0, Dl = params
    M, eta = mass_transform([m1, m2])
    return Psieff(fs, M, eta, t0=t0, phi0=phi0)

# get derivative of phase with respect to frequency
def get_phase_deriv(fs, params):
    m1, m2, t0, phi0, Dl = params
    M, eta = mass_transform([m1, m2])
    return dPsieff_df(fs, M, eta, t0=t0)


# get frequency-domain waveform
def get_waveform_freq(fs, params):
    return get_amp(fs, params) * np.exp(1.j * get_phase(fs, params))


# plot waveform at measured parameters
fs = np.linspace(20, 1000, 1000)
amps = get_amp(fs, ms_measured)
print(amps)
plt.loglog(fs, amps)
plt.show()

