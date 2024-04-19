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
def get_amp(fs, comp_masses):
    M, eta = mass_transform(comp_masses)
    return Aeff(fs, M, eta, DL)

def get_phase(fs, comp_masses, t0, phi0):
    M, eta = mass_transform(comp_masses)
    return Psieff(fs, M, eta, t0, phi0)

def get_phase_deriv(fs, comp_masses, t0):
    M, eta = mass_transform(comp_masses)
    return dPsieff_df(fs, M, eta, t0)


# plot waveform at measured parameters
fs = np.linspace(20, 1000, 1000)
amps = get_amp(fs, [m1_measured, m2_measured])
print(amps)
plt.loglog(fs, amps)
plt.show()
