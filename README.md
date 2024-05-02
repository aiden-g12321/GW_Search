# GW_Search
Search LIGO data for GW signal. Description of analysis in GW_search.pdf

## File descriptions

constants.py: Store constants for reference.

fit_template.py: This script fits a whitened template to whitened and bandpassed data in the time-domain.
The fit is so the chi-squared is minimized over the reference time, phase, and amplitude.

get_data.py: This script fetches open LIGO data and stores as .dat in data folder.

get_psd.py: This script estimates the PSD from data and returns interpolated PSD function.
It also defines a joint PSD for the two detectors.

GW_search.pdf: General description of analysis.

PhenomA.py: Script courtesy of Neil Cornish.
Source: https://github.com/eXtremeGravityInstitute/LISA_Sensitivity/blob/master/PhenomA.py

search.py: This script searches for GW candidate. It obtains the SNR time-series maximized over 
the template bank. It also computes the combined SNR^2 time-series and SNR(^2) histograms 
to estimate significance.

signal_tools.py: This script stores whitening and bandpassing methods.

SNR_series.py: This script computes the SNR time-series given a template bank and strain data.
It contains methods to compute the SNR series for a given array of data, for full 2 minutes of data,
or the series that is maximized over the template bank.

template_bank.py: This script calculates the coverage ellipses in parameter space, and defines a template bank.
It tests the template bank by making a histogram of mis-matches from random samples in parameter space.

waveform_tools.py: This script contains methods for waveform generation and operations such as inner product,
template metric, and mis-match.



