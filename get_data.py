import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from constants import *


def get_data(start_time, end_time, make_plots=False):
    
    # fetch data and times
    time_seriesH1 = TimeSeries.fetch_open_data('H1', start_time, end_time, sample_rate=4096)
    time_seriesL1 = TimeSeries.fetch_open_data('L1', start_time, end_time, sample_rate=4096)
    data_H1 = np.array(time_seriesH1)
    data_L1 = np.array(time_seriesL1)
    times = np.array(time_seriesH1.times)

    # save data to .txt file
    np.savetxt('times.txt', times)
    np.savetxt('data_H1.txt', data_H1)
    np.savetxt('data_L1.txt', data_L1)

    # plot data
    if make_plots:
        plt.plot(times, data_H1, label='Hanford')
        plt.plot(times, data_L1, label='Livingston')
        plt.legend(loc='upper right')
        plt.show()
        
    return


get_data(GPS_start-60, GPS_start, make_plots=True)
        

