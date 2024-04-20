import numpy as np
import matplotlib.pyplot as plt
from constants import *
from waveform_tools import *



def get_ellipse(freqs, params, Ss, df):
    
    # get eigenvectors of metric
    params = np.array(params)
    metric_proj = projected_metric(freqs, params, Ss, df)
    value, vector = np.linalg.eig(metric_proj)
    
    # set Dl component of vector to zero (numerical errors)
    vector[4,:] = 0

    # get length semi-major and semi-minor axis
    dx = 1.e-7
    axis_lengths = []
    for i in range(2):
        mismatch = 0.
        step = 0
        while mismatch < 0.05:
            step += vector[:,i] * dx
            mismatch = get_mismatch(metric_proj, params, params + step)
        axis_lengths.append(np.sqrt(np.inner(step, step)))
    
    semi_major_sec = max(axis_lengths)
    semi_minor_sec = min(axis_lengths)
    semi_major = semi_major_sec / MTSUN_SI
    semi_minor = semi_minor_sec / MTSUN_SI

    
    return


get_ellipse(fs, [m1_measured_sec, m1_measured_sec - 2*MTSUN_SI, 0., 0., 100.], psd, df)


