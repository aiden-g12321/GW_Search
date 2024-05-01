import numpy as np
import matplotlib.pyplot as plt
from SNR_series import *
from get_psd import *
from get_data import *
from signal_tools import *



# do GW candidate search using matched-filtering with template bank
def candidate_search(make_plots=False):

    # load PSDs
    H_psd, L_psd = individual_psds()

    # get maximum of SNR series over templates
    [H_params, L_params, H1_series, L1_series] = max_template_SNR(H_psd, L_psd)

    # check Hanford and Livingston max SNRs are with 10ms of one another
    H1_max_time = H1_series[0,list(H1_series[1]).index(max(H1_series[1]))]
    L1_max_time = L1_series[0,list(L1_series[1]).index(max(L1_series[1]))]
    time_delay = abs(H1_max_time - L1_max_time)
    
    # find time segment where max SNR lies
    start_times, stop_times = get_segment_times(GPS_start_time)
    for i in range(num_segments):
        if start_times[i] < H1_max_time < stop_times[i]:
            max_SNR_segment_index = i
        
    
    # check for candidacy
    if np.not_equal(H_params, L_params).any():
        print('H and L SNR series maximized at different templates!')
    if time_delay > max_time_delay:
        print('time delay > 10 milli-second')
    
    if make_plots:
        plt.subplot(2, 1, 1)
        plt.plot(H1_series[0], H1_series[1], label='Hanford')
        plt.axvline(GPS_event_time, color='red')
        plt.axvline(H1_max_time, color='green')
        plt.legend(loc='upper left')
        plt.subplot(2, 1, 2)
        plt.plot(L1_series[0], L1_series[1], label='Livingston', color='orange')
        plt.axvline(GPS_event_time, color='red')
        plt.axvline(L1_max_time, color='green')
        plt.legend(loc='upper left')
        plt.show()
    
    else:
        return [H_params, H1_max_time, L1_max_time, max_SNR_segment_index, H1_series, L1_series]



# plot SNR histograms per detector
def plot_SNR_hist(H1_series, L1_series):
    
    # get mean and standard deviation of SNR series
    H_mean = np.mean(H1_series[1])
    L_mean = np.mean(L1_series[1])
    H_std = np.std(H1_series[1])
    L_std = np.std(L1_series[1])
    
    # plot histogram
    plt.subplot(2, 1, 1)
    plt.hist(H1_series[1], bins=100, label='Hanford')
    plt.axvline(H_mean + H_std * 5, color='red', label=r'$\mu + 5\sigma$')
    plt.axvline(max(H1_series[1]), color='green', label='maximum SNR')
    plt.legend(loc='upper right')
    plt.xlabel('SNR')
    plt.subplot(2, 1, 2)
    plt.hist(L1_series[1], bins=100, label='Livingston')
    plt.axvline(L_mean + L_std * 5, color='red', label=r'$\mu + 5\sigma$')
    plt.axvline(max(L1_series[1]), color='green', label='maximum SNR')
    plt.legend(loc='upper right')
    plt.xlabel('SNR')
    plt.show()
    
    return


# get SNR-squared series (combined detectors)
def get_SNRsq(H1_series, L1_series):
    
    # initialize times and SNR-squared arrays
    times = H1_series[0]
    SNRsq = H1_series[1]**2
    
    dt = times[1] - times[0]
    n = len(times)
    
    # loop through every time
    for i in range(n):
        print(i / n)
        time = times[i]
        # get SNR series from Livingston +/- 10ms
        # keep_indices = np.where(np.bitwise_and(times > time - max_time_delay, times < time + max_time_delay))
        lower_index = max(0, int(i - max_time_delay/dt))
        upper_index = min(int(i + max_time_delay/dt), n)
        keep_indices = range(lower_index, upper_index)
        SNR_keep = np.array(L1_series[1])[keep_indices]
        # get maximum SNR from Livingston in this domain
        max_SNR_L = max(SNR_keep)
        # compute combined SNR-squared
        SNRsq[i] += max_SNR_L**2
    
    return [times, SNRsq]



# plot SNR-squared histograms (detectors combined)
def plot_SNRsq_hist(SNRsq):
    
    # get mean and standard deviation of SNR series
    mean = np.mean(SNRsq)
    st_dev = np.std(SNRsq)
    
    # plot histogram
    plt.hist(SNRsq, bins=100, label='combined SNR' + r'$^2$')
    plt.axvline(mean + st_dev * 5, color='red', label=r'$\mu + 5\sigma$')
    plt.axvline(max(SNRsq), color='green', label='maximum SNR' + r'$^2$')
    plt.legend(loc='upper right')
    plt.xlabel('SNR' + r'$^2$')
    plt.show()
    
    return





############################################################
###################### TESTING #############################
############################################################


# [max_params, H1_max_time, L1_max_time, max_SNR_segment_index, H1_series, L1_series] = candidate_search()

max_params = [1.87201150e-04, 1.77350168e-04, 0.00000000e+00, 0.00000000e+00, 1.02927125e+16]
max_SNR_segment_index = 8

times = np.loadtxt('data/times_' + str(max_SNR_segment_index) + '.dat')
H = np.loadtxt('data/H1_' + str(max_SNR_segment_index) + '.dat')
L = np.loadtxt('data/L1_' + str(max_SNR_segment_index) + '.dat')
H1_psd, L1_psd = individual_psds()
H_white_bp = get_white_bp(times, H, H1_psd)
L_white_bp = get_white_bp(times, L, L1_psd)
plt.subplot(2, 1, 1)
plt.plot(times, H_white_bp, label='Hanford')
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
plt.plot(times, L_white_bp, label='Livingston', color='orange')
plt.legend(loc='upper left')
plt.show()


# times, SNRsq = get_SNRsq(H1_series, L1_series)

# save SNR series for convenience
# np.savetxt('data/H_SNR.dat', H_series)
# np.savetxt('data/L_SNR.dat', L_series)

# save SNRsq series for convenience
# np.savetxt('data/times_SNRsq.dat', times)
# np.savetxt('data/combined_SNRsq.dat', SNRsq)

# load SNR series
H1_series = np.loadtxt('data/H_SNR.dat')
L1_series = np.loadtxt('data/L_SNR.dat')

# load SNR-squared series
times = np.loadtxt('data/times_SNRsq.dat')
SNRsq = np.loadtxt('data/combined_SNRsq.dat')

# plot_SNR_hist(H1_series, L1_series)
# plot_SNRsq_hist(SNRsq)



