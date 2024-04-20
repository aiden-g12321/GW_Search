import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import romb as integrate
from constants import *
from IMRPhenomD import AmpPhaseFDWaveform, IMRPhenomDGenerateh22FDAmpPhase
import IMRPhenomD_const as imrc


# get h22 (frequency-domain) object
# h22 includes amplitude and phase as array
def get_h22(params, freqs):

    # reference frequency
    MfRef_in = 0.
    
    # unpack parameters
    # input mass in solar masses, luminosity distance in mega-parsecs
    m1, m2, t0, phi0, Dl = params
    chi1 = 0.
    chi2 = 0.
    # convert masses to kg
    m1_SI =  m1*imrc.MSUN_SI
    m2_SI =  m2*imrc.MSUN_SI
    # convert luminosity distance to meters
    Dl_SI = Dl * 1.e6 * imrc.PC_SI

    # initialize amplitude, phase, times, and time derivative
    num_freqs = len(freqs)
    amp_imr = np.zeros(num_freqs)
    phase_imr = np.zeros(num_freqs)
    time_imr = np.zeros(num_freqs)
    timep_imr = np.zeros(num_freqs)

    #the first evaluation of the amplitudes and phase will always be much slower, because it must compile everything
    h22 = AmpPhaseFDWaveform(num_freqs,freqs,amp_imr,phase_imr,time_imr,timep_imr,0.,t0)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22,freqs,phi0,MfRef_in,m1_SI,m2_SI,chi1,chi2,Dl_SI)
    
    return h22


# get frequency-domain waveform for given parameters
def get_waveform_freq(params, freqs):
    h22 = get_h22(params, freqs)
    return h22.amp * np.exp(1.j * h22.phase)


fs = np.linspace(10, 1024, 2**9)
waveform = get_waveform_freq([m1_measured, m2_measured, 0, 0, 100], fs)
plt.loglog(fs, waveform)
plt.loglog(fs, abs(waveform))
plt.show()

