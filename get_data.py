'''This script fetches open LIGO data and stores as .dat in data folder.'''


import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from constants import *


# write times and strain data to .dat files
def save_data(start_time, end_time, file_name, make_plots=False):
    
    # fetch data and times
    time_seriesH1 = TimeSeries.fetch_open_data('H1', start_time, end_time, sample_rate=4096)
    time_seriesL1 = TimeSeries.fetch_open_data('L1', start_time, end_time, sample_rate=4096)
    data_H1 = np.array(time_seriesH1)
    data_L1 = np.array(time_seriesL1)
    times = np.array(time_seriesH1.times)

    # save data to .txt file
    np.savetxt('data/times_' + file_name + '.dat', times)
    np.savetxt('data/H1_' + file_name + '.dat', data_H1)
    np.savetxt('data/L1_' + file_name + '.dat', data_L1)

    # plot data
    if make_plots:
        plt.plot(times, data_H1, label='Hanford')
        plt.plot(times, data_L1, label='Livingston')
        plt.legend(loc='upper right')
        plt.show()
        
    return


# get start and stop GPS times for analysis chunks
def get_segment_times(event_start_time):
    start_times = []
    stop_times = []
    stop_time = event_start_time + 0.5
    for i in range(num_segments):
        start_times.append(stop_time - time_overlap)
        stop_times.append(start_times[i] + time_segment)
        stop_time = stop_times[i]
    return [start_times, stop_times]

print(get_segment_times(1242459797))

# # save data for psd estimation
# save_data(GPS_start_time-60., GPS_start_time, 'psd')

# # save data in number of segments
# start_times, stop_times = get_segment_times(GPS_start_time)
# for j in range(num_segments):
#     save_data(start_times[j], stop_times[j], str(j))

# # save full two minutes of data in one file
# save_data(GPS_start_time, GPS_start_time + 120., 'total', make_plots=True)

        

