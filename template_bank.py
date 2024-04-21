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
    m1 = mass_min_sec
    m2 = mass_min_sec - 2.*MTSUN_SI
    params = np.array([m1, m2, 0., 0., Dl100Mpc])
    # initialize arrays for bank
    paramss = [params]
    metrics = [projected_metric(freqs, params, Ss, df)]
    semi_major, semi_minor, angle = get_axes_angle(freqs, params, Ss, df)
    majors = [semi_major]
    minors = [semi_minor]
    angles = [angle]
    count = 0
    while params[0] < mass_max_sec:  # add to bank until m1 > 50*Msun
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
        plt.xlim(mass_min, mass_max)
        plt.ylim(mass_min, mass_max)
        plt.show()
    
    return [np.array(paramss), np.array(metrics)]



# test template bank by computing mis-match for random parameters
def test_bank(freqs, Ss, df, make_plots=True):
    
    # get template bank
    paramss, metrics = get_template_bank(freqs, Ss, df, make_plots=True)
    num_templates = len(paramss)
    
    # random draws in parameter space
    num_draws = int(1e4)
    m1s = np.random.uniform(mass_min_sec, mass_max_sec, num_draws)
    m2s = np.zeros(num_draws)
    for i in range(num_draws):
        m2s[i] = np.random.uniform(mass_min_sec, m1s[i])
    draws = np.array([[m1s[i], m2s[i], 0., 0., Dl100Mpc] for i in range(num_draws)])
    
    # for every draw, find the minimum mis-match from bank
    mismatches = np.zeros(num_draws)
    for i in range(num_draws):
        mismatches[i] = min([get_mismatch(metrics[j], paramss[j], draws[i]) for j in range(num_templates)])
    
    # plot histogram of mis-match values
    if make_plots:
        plt.hist(mismatches, bins=100, weights=np.ones(num_draws) / num_draws)
        plt.xlabel('mis-match')
        plt.ylabel('fraction of draws')
        plt.show()
    
    return mismatches




test_bank(fs, psd, df)


