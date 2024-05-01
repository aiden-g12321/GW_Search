'''Store constants for reference.'''


# event details
eventname = 'GW200311_115853'
GPS_start_time = 1267963091
GPS_event_time = 1267963151.3

# data segmenting parameters
num_segments = 17  # number of segments
time_segment = 8.  # seconds of each segment
time_overlap = 1.  # seconds of overlap

# unit conversions
MTSUN_SI = 4.925491025543575903411922162094833998e-6  # Solar mass in seconds
MSUN_SI = 1.988546954961461467461011951140572744e30  # Solar mass in kilograms
MRSUN_SI = 1.476625061404649406193430731479084713e3  # Solar mass in meters
CLIGHT = 2.99792458e8  # speed of light, m/s
PC_SI = 3.085677581491367278913937957796471611e16  # parsec in meters


# measured parameters
m1_measured = 34.2  # solar masses
m2_measured = 27.7  # solar masses
m1_measured_sec = m1_measured * MTSUN_SI  # seconds
m2_measured_sec = m2_measured * MTSUN_SI  # seconds


# 100 mega-parsecs in seconds
Dl100Mpc = 100. * 1.e6 * PC_SI / CLIGHT

# minimum and maximum masses considered in template bank
mass_min = 25.  # solar masses
mass_max = 50.  # solar masses
mass_min_sec = mass_min * MTSUN_SI  # seconds
mass_max_sec = mass_max * MTSUN_SI  # seconds

num_params = 5


# maximum time delay between detectors
max_time_delay = 0.010  # 10 milli-second


