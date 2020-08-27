"""
Package for smoothing data by using Fourier Transforms
Available functions: find_nearest_index, smooth_data
"""

import numpy as np


def find_nearest_index(searched_list, desired_value, res_limit=1e-4):
    """Returns the index of the 'searched_list' with a value close to
    'desired_value'.  The function stops once a value is reached that
    fulfills searched_list[index] - desired_value <= res_limit.

    :param searched_list: The list to be searched. Must be numerically ordered.
    :param desired_value: The value you want to find in the ordered list.
    :param res_limit: Maximum allowed difference between teh desired_value and the value of the list at the return index
    :return: index, searched_list[index]

    """
    if ((searched_list[0] > desired_value) or (searched_list[-1] < desired_value)):
        print('Desired value out of bounds')
        return -1, -1
    if desired_value == searched_list[0]:
        return 0, searched_list[0]
    if desired_value == searched_list[-1]:
        return len(searched_list) - 1, searched_list[-1]
    low_bound = 0
    high_bound = len(searched_list)
    res = 1e6
    while abs(res) > res_limit:
        guess_index = int((high_bound + low_bound)/2)
        guess_result = searched_list[guess_index]
        res = desired_value - guess_result
        if abs(res) > res_limit:
            if res > 0:
                low_bound = guess_index
            else:
                high_bound = guess_index
        else:
            return guess_index, searched_list[guess_index]


def smooth_data(xdata, ydata, low_frequency=0, high_frequency=-1):
    """Performs numpy.fft.rfft(ydata) and trims anything lower than
    'low_frequency' and anything higher than
    'high_frequency'. Frequencies are given in 1/cycles. Use
    high_frequency=-1 to only trim low frequencies.  Then performs
    inverse transform (numpy.fft.irfft) and returns the smoothed ydata
    set.  :param xdata: x values of data set.  :param ydata: y values
    of data set.  :param low_frequency: Trim frequencies 0 to
    low_frequency.  :param high_frequency: Trim frequencies
    high_frequency to highest frequency. -1 gives full range.
    :return: Smoothed ydata

    """
    fft_data = np.fft.rfft(ydata)
    fft_freqdata = np.fft.rfftfreq(len(ydata), d=xdata[1] - xdata[0])

    if high_frequency == -1:
        high_frequency = fft_freqdata[-1]

    freq_spacing = fft_freqdata[1] - fft_freqdata[0]
    low_freq_cutoff_index, _ = find_nearest_index(fft_freqdata, low_frequency, freq_spacing)
    high_freq_cutoff_index, _ = find_nearest_index(fft_freqdata, high_frequency, freq_spacing)

    modified_fft_data = np.array([(high_freq_cutoff_index >= n >= low_freq_cutoff_index) * x
                                 for n, x in enumerate(fft_data)])

    smoothed_ydata = np.fft.irfft(modified_fft_data, len(ydata))

    return smoothed_ydata
