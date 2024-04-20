import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from constants import *
from waveform_tools import *


# requiring mis-match < 0.05, we define an ellipse about every point in parameter space
# function gets lengths of axes in solar masses, and orientation angle
def get_axes_angle(freqs, params, Ss, df):
    
    # get eigenvectors of metric
    params = np.array(params)
    metric_proj = projected_metric(freqs, params, Ss, df)
    value, vector = np.linalg.eig(metric_proj)
    
    # machine precision sometimes generates (small) nonzero luminosity distance components of metric
    # to avoid runaways, set distance component of eigenvectors to 0.
    vector[4,:] = 0.

    # get length of semi-major and semi-minor axes
    # iterate along eigenvectors until mis-match >= 0.05
    dx = 1.e-7
    axis_lengths = []
    for i in range(2):
        mismatch = 0.
        step = 0
        while mismatch < 0.05:
            step += vector[:,i] * dx
            mismatch = get_mismatch(metric_proj, params, params + step)
        axis_lengths.append(np.sqrt(np.inner(step, step)))

    semi_major_sec = max(axis_lengths)  # seconds
    semi_minor_sec = min(axis_lengths)  # seconds
    semi_major = semi_major_sec / MTSUN_SI  # solar masses
    semi_minor = semi_minor_sec / MTSUN_SI  # solar masses
    
    # get angle orientation (in degrees)
    min_component = min(vector[:,0])
    max_component = max(vector[:,0])
    angle = np.arctan(min_component / max_component) * 180. / np.pi
    
    return [semi_major, semi_minor, angle]




# plot coverage of template bank
num_pts = 10
m1s_sec = np.linspace(25.*MTSUN_SI, 50.*MTSUN_SI, num_pts)
m2s_sec = np.array([m1 - 2.*MTSUN_SI for m1 in m1s_sec])
for i in range(num_pts):
    params =[m1s_sec[i], m2s_sec[i], 0., 0., 100.*1.e6*PC_SI/CLIGHT]
    semi_major, semi_minor, angle = get_axes_angle(fs, params, psd, df)
    m1sol, m2sol = convert_solar(params)
    e = Ellipse((m1sol, m2sol), 2*semi_major, 2*semi_minor, angle=angle, color='purple', alpha=0.6)
    a = plt.subplot()
    a.add_artist(e)
    plt.scatter(m1sol, m2sol, color='green')
plt.xlabel(r'$m_1\;\;(M_\odot)$')
plt.ylabel(r'$m_2\;\;(M_\odot)$')
plt.xlim(25, 50)
plt.ylim(25, 50)
plt.show()




