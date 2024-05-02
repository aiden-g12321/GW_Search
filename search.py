'''This script searches for GW candidate. It obtains the SNR time-series maximized over 
the template bank. It also computes the combined SNR^2 time-series and SNR(^2) histograms 
to estimate significance.
'''



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

    # save SNR series
    np.savetxt('data/H_SNR.dat', H1_series)
    np.savetxt('data/L_SNR.dat', L1_series)

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
        plt.legend(loc='upper left')
        plt.subplot(2, 1, 2)
        plt.plot(L1_series[0], L1_series[1], label='Livingston', color='orange')
        plt.legend(loc='upper left')
        plt.show()
    
    else:
        return [H_params, H1_max_time, L1_max_time, max_SNR_segment_index, H1_series, L1_series]



# plot SNR histograms per detector
def plot_SNR_hist():
    
    # load SNR series
    H1_series = np.loadtxt('data/H_SNR.dat')
    L1_series = np.loadtxt('data/L_SNR.dat')
    
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
def get_SNRsq(make_plots=False):
    
    # load SNR series
    H1_series = np.loadtxt('data/H_SNR.dat')
    L1_series = np.loadtxt('data/L_SNR.dat')
    
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
        lower_index = max(0, int(i - max_time_delay/dt))
        upper_index = min(int(i + max_time_delay/dt), n)
        keep_indices = range(lower_index, upper_index)
        SNR_keep = np.array(L1_series[1])[keep_indices]
        # get maximum SNR from Livingston in this domain
        max_SNR_L = max(SNR_keep)
        # compute combined SNR-squared
        SNRsq[i] += max_SNR_L**2
        
    # save SNR-squared series
    SNRsq_series = np.array([times, SNRsq])
    np.savetxt('data/SNRsq.dat', SNRsq_series)
    
    if make_plots:
        plt.plot(times, SNRsq)
        plt.xlabel('time (s)')
        plt.ylabel('SNR' + r'$^2$')
        plt.show()
    
    return SNRsq_series



# plot SNR-squared histograms (detectors combined)
def plot_SNRsq_hist():
    
    # get mean and standard deviation of SNR series
    times, SNRsq = np.loadtxt('data/SNRsq.dat')
    mean = np.mean(SNRsq)
    st_dev = np.std(SNRsq)
    
    significance = (max(SNRsq) - mean) / st_dev
    
    # plot histogram
    plt.hist(SNRsq, bins=100, label='combined SNR' + r'$^2$')
    plt.axvline(mean + st_dev * 5, color='red', label=r'$\mu + 5\sigma$')
    plt.axvline(max(SNRsq), color='green', label='maximum SNR' + r'$^2$')
    plt.legend(loc='upper right')
    plt.xlabel('SNR' + r'$^2$')
    plt.show()
    
    return significance



############################################################
###################### TESTING #############################
############################################################


# do search and get SNR series
# [max_params, H1_max_time, L1_max_time, max_SNR_segment_index, H1_series, L1_series] = candidate_search()
# times, SNRsq = get_SNRsq(make_plots=True)

# calculate amplitude ratio
H1_series = np.loadtxt('data/H_SNR.dat')
L1_series = np.loadtxt('data/L_SNR.dat')
print('amp ratio = ' + str(max(H1_series[1]) / max(L1_series[1])))

# plot SNR^2 time-series
times, SNRsq = np.loadtxt('data/SNRsq.dat')
plt.plot(times, SNRsq)
plt.axvline(GPS_event_time, color='red', alpha=0.5)
plt.xlabel('time (s)')
plt.ylabel('SNR' + r'$^2$')
plt.show()

# plot SNR histograms
plot_SNR_hist()
significance = plot_SNRsq_hist()
print('sig ~ ' + str(significance))

