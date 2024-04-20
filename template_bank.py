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
    
    # get eigenvector oriented in +m1 and +m2 direction
    if np.sign(vector[0,0]) == np.sign(vector[1,0]):
        eigen_vec = vector[:,0]
    else:
        eigen_vec = vector[:,1]
        
    # get angle (in degrees) of ellipse
    angle = np.arctan2(-eigen_vec[1], eigen_vec[0]) * 180. / np.pi
    
    return [semi_major, semi_minor, angle]


# get template bank, templates lie along m2 = m1 - 2*Msun line
# spacing determined so templates 0.05 mis-match from each other
def get_template_bank(freqs, Ss, df, make_plots=False):
    m1 = 25. * MTSUN_SI # start off at 25 solar masses
    m2 = m1 - 2. * MTSUN_SI
    params = np.array([m1, m2, 0., 0., 100.*1.e6*PC_SI/CLIGHT])
    # initialize arrays for bank
    paramss = [params]
    metrics = [projected_metric(freqs, params, Ss, df)]
    semi_major, semi_minor, angle = get_axes_angle(freqs, params, Ss, df)
    majors = [semi_major]
    minors = [semi_minor]
    angles = [angle]
    count = 0
    while params[0] / MTSUN_SI < 50.:  # add to bank until m1 > 50*Msun
        # move along m2 = m1 - 2*Msun line
        shift = minors[count]*MTSUN_SI / np.sqrt(2)
        params = paramss[count] + np.array([shift, shift, 0., 0., 0.])
        paramss.append(params)
        metrics.append(projected_metric(freqs, params, Ss, df))
        semi_major, semi_minor, angle = get_axes_angle(freqs, params, Ss, df)
        majors.append(semi_major)
        minors.append(semi_minor)
        angles.append(angle)
        count += 1
    
    # plot coverage of template bank
    if make_plots:
        a = plt.subplot(aspect='equal')
        for i in range(count):
            m1sol, m2sol = convert_solar(paramss[i])
            # scale width of ellipse by factor for clearer viewing
            factor = 3.
            e = Ellipse((m1sol, m2sol), 2.*majors[i], 2.*minors[i] / factor, 
                        angle=angles[i], color='purple', alpha=0.6)
            a.add_artist(e)
            plt.scatter(m1sol, m2sol, color='green')
        plt.xlabel(r'$m_1\;\;(M_\odot)$')
        plt.ylabel(r'$m_2\;\;(M_\odot)$')
        plt.xlim(25, 50)
        plt.ylim(25, 50)
        plt.show()
    
    return [np.array(paramss), np.array(metrics)]




paramss, metrics = get_template_bank(fs, psd, df, make_plots=True)
print(np.shape(paramss))
print(np.shape(metrics))




