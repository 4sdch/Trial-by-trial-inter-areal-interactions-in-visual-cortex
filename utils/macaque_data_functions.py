
from neo import NixIO
import pandas as pd
import numpy as np
import os
import glob
from scipy.ndimage import gaussian_filter1d
import h5py
import quantities as pq
#get working directory
main_dir = os.getcwd() + '/'


def get_epoch_times(resp_array, date, stim_on=0, stim_off=400, spont_stim_off =200, monkey='L', sampling_rate=1000):
    """
    This function reads metadata, processes epoch times, and extracts response and spontaneous activity
    arrays based on specified parameters.
    
    :param resp_array: The `resp_array` parameter in the `get_epoch_times` function is likely a numpy
    array containing neural responses data. This array is used to extract specific epochs of neural
    responses based on the provided parameters and return them as `true_resp` and `true_spont`
    :param date: The `date` parameter is used to specify the date for which you want to retrieve epoch
    times
    :param stim_on: The `stim_on` parameter in the `get_epoch_times` function represents the time (in
    milliseconds) when the stimulus starts in each epoch. By default, it is set to 0, but you can
    customize it if needed, defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `get_epoch_times` function represents the time in
    milliseconds when the stimulus ends during data processing. It is used to calculate the duration of
    the response window for each epoch, defaults to 400 (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_epoch_times` function represents
    the duration of spontaneous activity after the original stimulus onset. It is used to extract the
    spontaneous responses from the `resp_array` data. In the function, it is subtracted from the
    original stimulus onset time to, defaults to 200 (optional)
    :param monkey: The `monkey` parameter in the `get_epoch_times` function is used to specify which
    monkey's data to retrieve. It is a string parameter that indicates the monkey's name or identifier,
    defaults to L (optional)
    :return: The function `get_epoch_times` returns two arrays: `true_resp` and `true_spont`.
    `true_resp` contains responses data for each epoch with a specified stimulus duration, while
    `true_spont` contains spontaneous responses data for each epoch.
    """
    data_author = 'chen' if monkey in ['L','A'] else 'ponce'
    monkey_path_name = f'monkey_{monkey}' if monkey in ['L','A'] else monkey.lower()
    
    ms_to_samp = lambda ms: int(round(ms * sampling_rate / 1000.0)) # added this because lfp sampling rate is 500, not 1000 Hz.
    stim_on = ms_to_samp(stim_on)
    stim_off = ms_to_samp(stim_off)
    spont_stim_off = ms_to_samp(spont_stim_off)
    # get epoch times and convert to ms frames.
    df = pd.read_csv(main_dir + f'data/{data_author}/metadata/{monkey_path_name}/epochs_{monkey}_SNR_{date}.csv')
    new_df = df[df['success']].copy()
    # based on the sampling rate of 1000 or 500, we will index the array differently

    new_df['og_stim_on'] = (new_df['t_stim_on'] *sampling_rate).round().astype(int)
    new_df['new_stim_on'] = (new_df['t_stim_on'] *sampling_rate).round().astype(int) + stim_on
    # separate responses and make into a (epochs,frames_per_epoch, n_electrodes) array.
    # To make it more uniform, made all the epoch times last 400ms (ranges from 400ms to 410ms per epoch)

    resp_times = stim_off-stim_on
    
    true_resp = resp_array[new_df['new_stim_on'].values[:, None] + np.arange(resp_times), :] # treat values in ms from df as indices in array 
    true_spont = resp_array[(new_df['og_stim_on'] - int(spont_stim_off)).values[:, None] + np.arange(spont_stim_off), :]
    
    return true_resp, true_spont



def get_epoch_times_RF(resp_array, date, stim_on=0, stim_off=1000, spont_stim_off =200, monkey='L', direction_n=None,
                       sampling_rate=1000):
    """
    This function reads metadata from a CSV file, processes the data, and returns specific arrays based
    on input parameters for a moving bars dataset.
    
    :param resp_array: The `resp_array` parameter in the `get_epoch_times_RF` function is expected to be
    a NumPy array containing response data. This array likely represents neural responses to stimuli in
    a neuroscience experiment. The function processes this response data based on the other parameters
    provided to extract specific epochs of interest related to
    :param date: The `date` parameter in the `get_epoch_times_RF` function is used to specify the date
    for which you want to retrieve epoch times and convert them to milliseconds frames for the moving
    bars dataset
    :param stim_on: The `stim_on` parameter in the `get_epoch_times_RF` function represents the time (in
    milliseconds) when the stimulus starts during the experiment. It is used to calculate the new
    stimulus onset time based on the original stimulus onset time in the dataset, defaults to 0
    (optional)
    :param stim_off: The `stim_off` parameter in the `get_epoch_times_RF` function represents the time
    in milliseconds when the stimulus ends during data processing for the moving bars dataset. It is
    used to calculate the duration of the response window after the stimulus onset (`stim_on`) and to
    extract the corresponding response data from, defaults to 1000 (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_epoch_times_RF` function
    represents the duration in milliseconds after the original stimulus onset time (`og_stim_on`) for
    which you want to capture responses for the spontaneous activity. It is used to define the time
    window for extracting responses before the, defaults to 200 (optional)
    :param monkey: The `monkey` parameter in the `get_epoch_times_RF` function is used to specify the
    monkey for which the data is being processed. It is a string parameter that indicates the name of
    the monkey (e.g., 'L' for monkey L). This parameter is used to load the corresponding metadata,
    defaults to L (optional)
    :param direction_n: The `direction_n` parameter specifies whether to retrieve only one direction of
    sweeping bar. The options are integers from 0 to 3. If you provide a value for `direction_n`, the
    function will filter the data to retrieve responses only for that specific direction of sweeping bar
    :return: The function `get_epoch_times_RF` returns three values: `true_resp`, `true_spont`, and
    `cond_labels`.
    """
    # get epoch times and convert to ms frames for moving bars dataset.
    
    ms_to_samp = lambda ms: int(round(ms * sampling_rate / 1000.0)) # added this because lfp sampling rate is 500, not 1000 Hz.
    stim_on = ms_to_samp(stim_on)
    stim_off = ms_to_samp(stim_off)
    spont_stim_off = ms_to_samp(spont_stim_off)
    
    df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_RF_{date}.csv')
    new_df = df[df['success']].copy()
    new_df['og_stim_on'] = (new_df['t_stim_on'] *sampling_rate).round().astype(int)
    new_df['new_stim_on'] = (new_df['t_stim_on'] *sampling_rate).round().astype(int) + stim_on
    
    mapping = {}
    for count, label in enumerate(new_df['cond'].unique()):
        mapping[label]=count
    mapping
    new_df['cond_num']=new_df['cond'].map(mapping)

    resp_times = stim_off-stim_on

    if direction_n is not None:
        ## specifies whether to retrieve only one direction of sweeping bar. options are 0 to 3.
        query = f'cond_num == {direction_n}'
        true_resp = resp_array[new_df.query(query)['new_stim_on'].values[:, None] + np.arange(resp_times), :]
        true_spont = resp_array[(new_df.query(query)['og_stim_on'] - int(spont_stim_off)).values[:, None] + np.arange(spont_stim_off), :]
    else:
        true_resp = resp_array[new_df['new_stim_on'].values[:, None] + np.arange(resp_times), :]
        true_spont = resp_array[(new_df['og_stim_on'] - int(spont_stim_off)).values[:, None] + np.arange(spont_stim_off), :]
    
    labels = new_df.cond_num.values
    cond_labels = np.tile(labels,(resp_times,1)).T
    cond_labels= cond_labels[:,:,np.newaxis]

    # print(f'true_resp shape: {true_resp.shape}, true_spont shape: {true_spont.shape}, cond_labels shape: {cond_labels.shape}')
    return true_resp, true_spont, cond_labels


def get_number_of_epochs(date, monkey='L', condition_type='SNR'):
    """   This function retrieves the number of epochs for a specific date, monkey, and condition type from a
    CSV file.

    Args:
        date (_type_): The `date` parameter in the `get_number_of_epochs` function is used to specify the date
    for which you want to retrieve the number of epochs
        monkey (str, optional): The `monkey` parameter in the `get_number_of_epochs` function is used to specify the
    monkey for which you want to retrieve the number of epochs. By default, it is set to 'L', defaults
    to L (optional). Defaults to 'L'.
        condition_type (str, optional): It is a string parameter that specifies the condition type, such as 'SNR'
    (Signal-to-Noise Ratio) in this case, defaults to SNR (optional)
    :return: the number of epochs that meet the 'success' condition in the provided DataFrame after
    reading and filtering the data from a CSV file. Defaults to 'SNR'.

    Returns:
        _type_: _description_
    """
    # get epoch times and convert to ms frames.
    df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_{condition_type}_{date}.csv')
    new_df = df[df['success']].copy()
    return len(new_df)


def bin_labels(data, window_size, **kwargs):
    """
    The function `bin_labels` takes a dataset and aggregates the data points into bins of a specified
    window size.
    
    :param data: Data is the input array containing the datapoints that need to be binned. It is a 2D
    numpy array where each row represents a datapoint and each column represents a feature or dimension
    of the datapoint
    :param window_size: The `window_size` parameter in the `bin_labels` function represents the desired
    size of each window for binning the data points. This value determines how many data points will be
    grouped together and processed at a time during the binning process
    :return: The function `bin_labels` returns binned data where the datapoints from the input data are
    aggregated into the desired window size using the median function. The binned data is returned as a
    numpy array with dimensions [bin_datapoints, number of columns in the input data].
    """
    ## bings the datapoints from 1ms to desired window size.

    bin_datapoints = int(np.floor(len(data)/window_size))
    binned_data = np.zeros([bin_datapoints, data.shape[1]])
    for i in range(bin_datapoints):
        start = int(round(i * window_size))
        end   = int(round((i + 1) * window_size))
        window_data = data[start:end, :]
        binned_data[i] = np.median(window_data, axis=0)
        
    return binned_data



def isolate_norm_resps_RF(resp_array, date='260617', monkey='L', bin_function=None, stim_on=0, 
                          stim_off=1000, raw_resp=False, return_binned=False, sampling_rate=1000,**kwargs):
    """
    This function isolates responses for moving bars by removing gray screen presentation activity and
    optionally normalizes the responses.
    
    :param resp_array: The `resp_array` parameter is a numpy array containing neural responses
    data. This function processes this data to isolate responses for moving bars by removing gray
    screen presentation activity. The function also allows for binning the responses using a specified
    function, normalizing the responses, and returning the normalized responses
    :param date: The `date` parameter in the `isolate_norm_resps_RF` function is used to specify the
    date for which the responses are being analyzed. It has a default value of '260617' if not provided
    explicitly, defaults to 260617 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_resps_RF` function is used to specify the
    monkey from which the responses were recorded. It is a string parameter that indicates the monkey's
    identity, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_resps_RF` function is a
    function that can be applied to each epoch response in order to bin or process the data in a
    specific way. This function should take an epoch response as input and return the processed or
    binned version of that
    :param stim_on: The `stim_on` parameter in the `isolate_norm_resps_RF` function specifies the time
    point at which the stimulus starts during the experiment. It is used to determine the beginning of
    the stimulus presentation period for isolating responses to moving bars in the `resp_array`,
    defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `isolate_norm_resps_RF` function specifies the time
    point at which the stimulus ends during data processing. It is used to define the duration of the
    stimulus presentation window. In the provided function, the default value for `stim_off` is set to,
    defaults to 1000 (optional)
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_resps_RF` function is a boolean flag
    that determines whether to return the raw responses or the normalized responses. If `raw_resp` is
    set to `True`, the function will return the true responses without normalization. If `raw_resp`,
    defaults to False (optional)
    :return: If the `raw_resp` parameter is `True`, the function will return the `true_resp` array and
    `cond_labels`. If `raw_resp` is not `True`, the function will return the `norm_resp` array and
    `binned_labels`.
    """
    ### removes the gray screen presentation activity to only obtain isolated responses for moving bars 
    true_resp, true_spont, cond_labels = get_epoch_times_RF(resp_array, stim_on=stim_on, stim_off=stim_off, 
                                                            date=date, monkey=monkey, sampling_rate=sampling_rate)
    
    if bin_function is not None:
        binned_resp = np.stack([bin_function(epoch_resp, **kwargs) for epoch_resp in true_resp])
        binned_spont = np.stack([bin_function(epoch_spont,**kwargs) for epoch_spont in true_spont])
        binned_labels = np.stack([bin_labels(epoch_label, **kwargs) for epoch_label in cond_labels])
    else:
        binned_resp = true_resp
        binned_spont = true_spont
        binned_labels=cond_labels
    
    if raw_resp is True:
        if return_binned  is True:
            return true_resp, cond_labels
        else:
            return binned_resp.reshape(-1, resp_array.shape[1]), binned_labels
        
    #5oct2025 i want to add the option of returning binned_resp with raw_resp=False
    
    else:
        norm_resp = binned_resp - np.mean(binned_spont, axis=1, keepdims=True)  
        if return_binned is True:
            # print(f'returning binned_resp shape: {binned_resp.shape}, binned_labels shape: {binned_labels.shape}')
            return binned_resp, binned_labels
        
        norm_resp = norm_resp.reshape(-1, resp_array.shape[1])
        # norm_resp -= np.mean(norm_resp, axis=0)
        
        # print(norm_resp.shape)
        return norm_resp, binned_labels

def isolate_norm_spont_RF(resp_array, date='260617', monkey='L', bin_function=None, stim_on=0, stim_off=1000,raw_resp=False,
                          spont_stim_off=300, return_binned=False, sampling_rate=1000, **kwargs):
    """
    This Python function isolates responses for gray screen activity by removing moving bars activity
    and optionally binning the data.
    
    :param resp_array: `resp_array` is likely a numpy array containing responses from a neural recording
    experiment. The function `isolate_norm_spont_RF` processes this array to isolate responses to gray
    screen activity by removing moving bars activity
    :param date: The `date` parameter in the `isolate_norm_spont_RF` function is used to specify the
    date for which the responses are being analyzed. It has a default value of '260617' if not provided
    explicitly when calling the function, defaults to 260617 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_spont_RF` function is used to specify the
    monkey from which the response data is obtained, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_spont_RF` function is used to
    specify a function that will be applied to each epoch of spontaneous activity data before further
    processing. This function can be used to perform operations such as binning, averaging, or any other
    data transformation on the
    :param stim_on: The `stim_on` parameter in the `isolate_norm_spont_RF` function specifies the time
    point at which the stimulus starts during the recording. It is used to define the beginning of the
    stimulus presentation window for isolating responses related to the gray screen activity, defaults
    to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `isolate_norm_spont_RF` function represents the
    time point at which the stimulus ends during the experiment. It is used to define the duration of
    the stimulus presentation period. In this function, it is set to a default value of 1000, defaults
    to 1000 (optional)
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_spont_RF` function is a boolean flag
    that determines whether to return the raw spontaneous responses or not. If `raw_resp` is set to
    `True`, the function will return the raw spontaneous responses along with the condition labels. If
    `, defaults to False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `isolate_norm_spont_RF` function
    represents the duration (in milliseconds) after the stimulus offset where spontaneous activity is
    considered. In this function, it is used to determine the time window for isolating spontaneous
    responses after the stimulus presentation has ended, defaults to 300 (optional)
    :return: The function `isolate_norm_spont_RF` returns the normalized spontaneous responses and
    condition labels after isolating the gray screen activity from the input response array.
    """
    ### removes the moving bars activity to only obtain isolated responses for gray screen activity 
    true_resp, true_spont, cond_labels = get_epoch_times_RF(resp_array, stim_on=stim_on, stim_off=stim_off, date=date, monkey=monkey, spont_stim_off=spont_stim_off, sampling_rate=sampling_rate)
    
    if bin_function is not None:
        binned_spont = np.stack([bin_function(epoch_spont,**kwargs) for epoch_spont in true_spont])
        binned_labels = np.stack([bin_labels(epoch_label, **kwargs) for epoch_label in cond_labels])
    else:
        binned_spont = true_spont
        binned_labels=cond_labels
    
    
    if raw_resp is True:
        if return_binned  is True:
            return true_resp, cond_labels
        else:
            return binned_spont.reshape(-1, resp_array.shape[1]), cond_labels
    if return_binned is True: #5oct2025 i want to add the option of returning binned_resp with raw_resp=False
        return binned_spont, binned_labels
        
    norm_spont = binned_spont.reshape(-1, resp_array.shape[1])

    return norm_spont, binned_labels

def isolate_norm_resps(resp_array, date='250717', monkey='L', bin_function=None, stim_on=0, 
                       stim_off=400, shuffle=False, seed=None, raw_resp=False, 
                       return_binned=False, sampling_rate=1000,**kwargs):
    """
    This function isolates responses for checkerboard presentations by removing gray screen activity and
    optionally performs trial shuffling or normalization.
    
    :param resp_array: `resp_array` is an array containing neural responses data, typically in the form
    of responses to checkerboard presentations
    :param date: The `date` parameter in the `isolate_norm_resps` function is used to specify the date
    for which the responses are being analyzed. It is set to a default value of '250717' but can be
    changed to any specific date for which you want to isolate responses, defaults to 250717 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_resps` function is used to specify the
    monkey from which the responses were recorded, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_resps` function allows you to
    specify a function that will be applied to each epoch response before further processing. This can
    be useful for binning or aggregating the data in a specific way before normalization or analysis
    :param stim_on: The `stim_on` parameter in the `isolate_norm_resps` function specifies the time
    point at which the stimulus presentation begins. It is used to define the start of the epoch for
    extracting responses to the checkerboard presentations, defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `isolate_norm_resps` function represents the end
    time of the stimulus presentation in milliseconds. It is used to define the duration of the stimulus
    presentation window for isolating responses to checkerboard presentations, defaults to 400
    (optional)
    :param shuffle: The `shuffle` parameter in the `isolate_norm_resps` function allows you to specify
    whether you want to perform trial shuffling of checkerboard images before processing the responses.
    If `shuffle` is set to `True`, the function will shuffle the order of the responses using random
    indices before further, defaults to False (optional)
    :param seed: The `seed` parameter in the `isolate_norm_resps` function is used to set the random
    seed for reproducibility when shuffling the indices of the responses. By setting a specific seed
    value, you can ensure that the same random shuffling is applied each time the function is called
    with
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_resps` function is a boolean flag
    that determines whether to return the raw responses without normalization or not. If `raw_resp` is
    set to `True`, the function will return the binned responses reshaped as a 2D array, defaults to
    False (optional)
    :return: The function `isolate_norm_resps` returns either the reshaped binned responses or the
    normalized responses based on the input parameters. If `raw_resp` is True, it returns the binned
    responses reshaped as a 2D array. If `raw_resp` is False, it returns the normalized responses
    calculated as the binned responses minus the mean of the binned spontaneous responses, resh
    """
    ### removes the gray screen presentation activity to only obtain isolated responses for checkerboard presentations
    true_resp, true_spont = get_epoch_times(resp_array, stim_on=stim_on, stim_off=stim_off, 
                                            date=date, monkey=monkey, sampling_rate=sampling_rate)
    
    # print('before binning resp_array shape:', resp_array.shape)
    if bin_function is not None:
        binned_resp = np.stack([bin_function(epoch_resp, **kwargs) for epoch_resp in true_resp])
        binned_spont = np.stack([bin_function(epoch_spont, **kwargs) for epoch_spont in true_spont])
        # print('binned_resp shape:', binned_resp.shape)
    else:
        binned_resp = true_resp
        binned_spont = true_spont

    if shuffle is True:
        # to perform trial shuffling of checkerboard images
        indices = np.arange(len(binned_resp)) 
        # Shuffle the indices using np.random.shuffle
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        binned_resp = binned_resp[indices]
        binned_spont = binned_spont[indices]

    if raw_resp is True:
        if return_binned  is True:
            return true_resp
        else:
            return binned_resp.reshape(-1, resp_array.shape[1]) #resp_array.shape[1] is the number of electrodes
    
    if binned_resp.shape[2]==0:
        return None
    norm_resp = binned_resp - np.mean(binned_spont, axis=1, keepdims=True)
    #05oct2025 i want to add the option of returning binned_resp with raw_resp=False
    if return_binned is True:
        return binned_resp
    norm_resp = norm_resp.reshape(-1, resp_array.shape[1])#resp_array.shape[1] is the number of electrodes

    return norm_resp

def isolate_norm_spont(resp_array, date='250717', monkey='L', bin_function=None, shuffle=False, 
                       seed=None, raw_resp=False,spont_stim_off=300, 
                       return_binned=False, sampling_rate=1000, **kwargs):
    """
    This Python function isolates and normalizes spontaneous responses from an array of responses, with
    options for binning, shuffling, and reshaping the data.
    
    :param resp_array: `resp_array` is a numpy array containing the response data
    :param date: The `date` parameter in the function `isolate_norm_spont` is used to specify the date
    for which the response array should be analyzed. It has a default value of '250717' if not provided
    explicitly, defaults to 250717 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_spont` function is used to specify the
    monkey from which the response data is coming, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_spont` function allows you to
    specify a function that will be applied to each epoch of spontaneous activity data before further
    processing. This function should take an epoch of spontaneous activity data as input and return the
    processed output
    :param shuffle: The `shuffle` parameter in the `isolate_norm_spont` function is a boolean flag that
    determines whether the data should be shuffled before processing. If `shuffle` is set to `True`, the
    function will shuffle the data using random indices before further processing. This can be useful
    for randomizing, defaults to False (optional)
    :param seed: The `seed` parameter in the `isolate_norm_spont` function is used to set the random
    seed for reproducibility when shuffling the indices of the binned spontaneous responses. If a
    specific `seed` value is provided, it will ensure that the same random shuffling of indices is
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_spont` function is a boolean flag
    that determines whether the function should return the binned spontaneous responses reshaped as a 2D
    array or as a 1D array, defaults to False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `isolate_norm_spont` function is used
    to specify the time offset for spontaneous activity stimulation. This parameter determines the time
    at which the spontaneous activity ends. In the function, it is set to a default value of 300,
    defaults to 300 (optional)
    :return: The function `isolate_norm_spont` returns the normalized spontaneous responses based on the
    input parameters and conditions specified in the function. If `raw_resp` is True, it returns the
    reshaped binned spontaneous responses. If `shuffle` is True, it shuffles the indices before
    reshaping the binned spontaneous responses. Finally, it returns the normalized spontaneous responses
    in the shape (-1, resp
    """
    
    true_resp, true_spont = get_epoch_times(resp_array, date, spont_stim_off=spont_stim_off, monkey=monkey, sampling_rate=sampling_rate)

    
    if bin_function is not None:
        binned_spont = np.stack([bin_function(epoch_spont, **kwargs) for epoch_spont in true_spont])
    else:
        binned_spont = true_spont
        
    if raw_resp is True:
        if return_binned is True:
            return true_spont
        else:
            return binned_spont.reshape(-1, resp_array.shape[1])
    
    if shuffle is True:
        indices = np.arange(len(binned_spont)) 
        # Shuffle the indices using np.random.shuffle
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        binned_spont = binned_spont[indices]
    
    if binned_spont.shape[2]==0:
        return None
    if return_binned is True: #5oct2025 i want to add the option of returning binned_resp with raw_resp=False
        return binned_spont
    norm_spont = binned_spont.reshape(-1, resp_array.shape[1])
    return norm_spont

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Tuple, List
def isolate_RS_resp(
    resp_array: np.ndarray,
    date: str,
    open_or_closed: str = "Open_eyes",
    monkey: str = "L",
    main_dir: str = ".",
    bin_function:  Optional[Callable[..., np.ndarray]] = None,
    t0_ms: int = 0,  # ms offset if your resp_array starts after absolute 0
    sampling_rate: int = 1000,  # Hz
    **kwargs,
):
    """
    Select RS epochs by eye state and return:
      - resp_sel: concatenated samples (after per-epoch binning if provided)
      - mask: boolean mask over resp_array (True = selected)
      - slices: list of slice(start, stop) actually used (stop exclusive)
      - info: dict with diagnostics (counts, overlaps, gaps)
    """
    
    t0_sample = int(t0_ms * sampling_rate / 1000)
    # 1) Load metadata
    if monkey in ["D"]:
        meta_path = Path(main_dir) / f"data/ponce/metadata/{monkey}/epochs_{monkey}_RS_{date}.csv"
    else:
        meta_path = Path(main_dir) / f"data/chen/metadata/monkey_{monkey}/epochs_{monkey}_RS_{date}.csv"
    df = pd.read_csv(meta_path)

    # 2) Filter eye state and coerce numeric
    df = df.query('state == @open_or_closed').copy()
    for col in ("t_start", "t_stop", "dur"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3) Convert samples
    df["t_start_"] = np.round(df["t_start"] * sampling_rate).astype("Int64")
    df["dur_"]     = np.round(df["dur"]     * sampling_rate).astype("Int64")
    df["t_stop_"]  = df["t_start_"] + df["dur_"]  # stop exclusive by construction

    # 4) Shift into resp_array index space by t0_ms, and sort
    df["idx_start"] = (df["t_start_"] - t0_sample).astype("Int64")
    df["idx_stop"]  = (df["t_stop_"]  - t0_sample).astype("Int64")
    df = df.sort_values("idx_start")

    T = resp_array.shape[0]
    mask = np.zeros(T, dtype=bool)
    slices: List[slice] = []
    clipped = 0

    # 5) Build non-overlapping, clipped slices (stop exclusive)
    for s, e in zip(df["idx_start"].to_numpy(), df["idx_stop"].to_numpy()):
        if pd.isna(s) or pd.isna(e):
            continue
        s = int(s); e = int(e)
        s_clip = max(0, s)
        e_clip = min(T, e)
        if e_clip > s_clip:
            # mark mask and record slice
            mask[s_clip:e_clip] = True
            slices.append(slice(s_clip, e_clip))
            if (s_clip != s) or (e_clip != e):
                clipped += 1

    # 6) Concatenate data, optionally with per-epoch binning
    if bin_function is None:
        resp_sel = resp_array[mask]
    else:
        chunks = []
        for sl in slices:
            arr = resp_array[sl]
            out = bin_function(arr, **kwargs)
            if out.size:
                chunks.append(out)
        resp_sel = np.concatenate(chunks, axis=0) if chunks else np.array([])

    # 7) Diagnostics: overlaps/gaps after conversion
    starts = np.array([sl.start for sl in slices])
    stops  = np.array([sl.stop  for sl in slices])
    overlaps = (starts[1:] < stops[:-1]).sum() if len(slices) > 1 else 0
    gaps     = (starts[1:] > stops[:-1]).sum() if len(slices) > 1 else 0

    info = dict(
        total_T=T,
        selected_mask_sum=int(mask.sum()),
        n_epochs=len(slices),
        n_clipped=clipped,
        n_overlaps=int(overlaps),
        n_gaps=int(gaps),
    )
    return resp_sel


from scipy.stats import sem
def get_img_resp_avg_sem(resp_array, condition_type, w_size=25,chunk_size=None,
                        get_chunks=False, step_size=None, sampling_rate=1000):
    """
    This Python function calculates the average and standard error of the mean of neural responses based
    on input parameters such as response array, date, condition type, window size, and chunk size.
    
    :param resp_array: `resp_array` is a numpy array containing responses from neurons. The function
    `get_img_resp_avg_sem` processes this array along with other parameters to calculate the average and
    standard error of the mean (SEM) of neuron responses based on the specified conditions
    :param condition_type: Condition type refers to the type of experimental condition or stimulus being
    presented to the neurons. It could include conditions like 'RF_spont', 'RF', 'SNR_spont', 'SNR',
    etc. These conditions may have different chunk sizes based on the type
    :param w_size: The `w_size` parameter in the `get_img_resp_avg_sem` function represents the window
    size used for calculating the chunk size based on the condition type. It is used to determine the
    number of frames in each chunk based on the condition type specified, defaults to 25 (optional)
    :param chunk_size: The `chunk_size` parameter in the `get_img_resp_avg_sem` function determines the
    size of each chunk that the response array will be split into. The size of the chunks is calculated
    based on the `condition_type` provided. If `chunk_size` is not explicitly provided, it is calculated
    :param get_chunks: The `get_chunks` parameter in the `get_img_resp_avg_sem` function is a boolean
    flag that determines whether the function should return the chunks_array or not. If `get_chunks` is
    set to `True`, the function will return the chunks_array. If `get_chunks` is set to, defaults to
    False (optional)
    :return: the average neuron response and the standard error of the mean (SEM) of the neuron
    response.
    """
    ms_to_samp = lambda ms: int(round(ms * sampling_rate / 1000.0)) # added this because lfp sampling rate is 500, not 1000 Hz.

    if chunk_size is None and step_size is None:
        if 'RF_spont' in condition_type:
            chunk_size=int(ms_to_samp(200)/w_size)
        
        elif 'RF' in condition_type:
            chunk_size=int(ms_to_samp(1000)/w_size)
        
        elif 'SNR_spont' in condition_type:
            chunk_size=int(ms_to_samp(200)/w_size)

        elif 'SNR' in condition_type:
            chunk_size=int(ms_to_samp(400)/w_size)

    elif step_size is not None:
        if 'RF_spont' in condition_type:
            chunk_size=int(np.floor((ms_to_samp(200) - w_size) / step_size)) + 1
        
        elif 'RF' in condition_type:
            chunk_size=int(np.floor((ms_to_samp(1000) - w_size) / step_size)) + 1
        
        elif 'SNR_spont' in condition_type:
            chunk_size=int(np.floor((ms_to_samp(200) - w_size) / step_size)) + 1

        elif 'SNR' in condition_type:
            chunk_size=int(np.floor((ms_to_samp(400) - w_size) / step_size)) + 1
        else:
            raise ValueError(f"Condition type {condition_type} not recognized")
            
    n_frames = resp_array.shape[0]
    n_epochs = int(n_frames/chunk_size)

    if not is_factor(chunk_size, n_frames):
        print('Frames are not evenly split among epochs')
        del chunk_size
    chunks = np.split(resp_array[:n_epochs * chunk_size, :], n_epochs)
    chunks_array = np.array(chunks)

    if get_chunks is True:
        return chunks_array
    
    avg_neuron_resp = np.mean(chunks_array, axis=0)
    SEM_neuron_resp = sem(chunks_array, axis=0)
    return avg_neuron_resp.T, SEM_neuron_resp.T
    

def is_factor(number, n):
    return n % number == 0


def binning_with_sum(data, window_size, e=0, **kwargs):
    # made it round of window_size because sometimes window_size is not an integer
    import numpy as np
    n_samples, n_features = data.shape
    # Number of bins based on full coverage
    n_bins = int(np.floor(n_samples / window_size))
    binned_data = np.zeros((n_bins, n_features))

    for i in range(n_bins):
        start = int(round(i * window_size))
        end   = int(round((i + 1) * window_size))
        binned_data[i] = np.sum(data[start:end, :], axis=0)

    return binned_data

def binning_with_avg(data, window_size, e=0, **kwargs):
    # made it round of window_size because sometimes window_size is not an integer
    import numpy as np
    n_samples, n_features = data.shape
    # Number of bins based on full coverage
    n_bins = int(np.floor(n_samples / window_size))
    binned_data = np.zeros((n_bins, n_features))

    for i in range(n_bins):
        start = int(round(i * window_size))
        end   = int(round((i + 1) * window_size))
        binned_data[i] = np.mean(data[start:end, :], axis=0)

    return binned_data


def get_resps(monkey, **args):
    if monkey in ['L', 'A']:
        return get_resps_chen(monkey=monkey, **args)
    elif monkey in ['D']:
        return get_resps_ponce(monkey=monkey, **args)
    else:
        raise ValueError(f"Monkey {monkey} not recognized")

ponce_area_elec_dict = {}
ponce_area_elec_dict['D']= {'V1':np.arange(1,17), 'V4':np.arange(17,33), 'IT':np.arange(33,65)}


def get_resps_ponce(condition_type='SNR', date='250225', monkey='D', w_size = 25, stim_on=0,  stim_off=400, 
            shuffle=False, bin_function=binning_with_sum, keep_SNR_elecs=False, 
            raw_resp=False, spont_stim_off=300, return_binned=False, SNR_threshold=2, 
            seed=None, calculate_SNRs=False, calculate_relis=False,
            step_size=10, subtract_spont_resp=True,
            activity_type='MUAe', 
            sampling_rate=1000, return_area_df=False, 
            z_score =True, **args):
    w_eff_ponce = int(round(w_size))  # Ponce keeps fs=1000 → 25 ms == 25 samples

    if 'SNR' in condition_type:
        array_condition_type = 'SNR'
    elif 'RS' in condition_type:
        array_condition_type = 'RS'
    else:
        array_condition_type = condition_type

    act_type_df = '_MUAe'
    
    # set the data directory and paths    
    data_dir = os.path.join(main_dir, f'data/ponce/{monkey}/{date}/{array_condition_type}')
    neural_data_path = glob.glob(data_dir + '/*MUAe.nix')[0]
    meta_path = glob.glob(f'data/ponce/{monkey}/{date}/SNR/*spikeMeta*.csv')[0]
    snr_data_path = os.path.join(main_dir, f'data/ponce/metadata/{monkey}/{monkey}_{date}_SNR{act_type_df}.csv')
    
    ##### load metadata and get rid of artifact units############
    area_df= pd.read_csv(meta_path) 
    
    # assign area names to the spikeIDs in area_df
    area_df['area'] = ''
    for area, elec_nums in ponce_area_elec_dict[monkey].items():
        area_df.loc[area_df['spikeID'].isin(elec_nums), 'area'] = area
    area_df['index'] = area_df.index
    print(f'length of area_df indices: {len(area_df)}')
    area_df = area_df.drop_duplicates(subset=['spikeID'])
    area_df['unitID'] = 1 # since MUAe activity did not separate to single units, assign all unitIDs to 1
    area_df = area_df.reset_index(drop=True)
    
    #creat unique spikeID for each electrod
    V4_indices = area_df[area_df['area']=='V4']['spikeID'].values - 1 #MUAe activity did not separate to single units. also change to -1 for zero indexing
    V1_indices = area_df[area_df['area']=='V1']['spikeID'].values - 1

        
    print(f'Number of V4 electrodes before filtering: {len(V4_indices)}')
    print(f'Number of V1 electrodes before filtering: {len(V1_indices)}')
    
    
    # check whether there is a snr_data_path and filter V1 and V4 indices accordingly
    if not keep_SNR_elecs:
        print('Filtering electrodes by SNR threshold of:', SNR_threshold)
        assert os.path.exists(snr_data_path), f'No SNR data found for {monkey} on {date}'
        snr_df = pd.read_csv(snr_data_path) 
        # # since the snr_df also may have artifacts i need to omit them also 
        snr_df = snr_df[snr_df['SNR']>= SNR_threshold]
        # also get rid of inf values and print if there are any
        if np.isinf(snr_df['SNR']).any() or np.isneginf(snr_df['SNR']).any():
            print(f'Found -inf values in SNR data for {monkey} on {date}. Electrode/s {snr_df[snr_df["SNR"].isin([np.inf, -np.inf])]["index"].values[0]}. Removing them.')
        snr_df = snr_df[~snr_df['SNR'].isin([np.inf, -np.inf])]

        V4_indices = [i for i in V4_indices if i in snr_df['index'].values]
        V1_indices = [i for i in V1_indices if i in snr_df['index'].values]

    
    #####################################################################################
    if array_condition_type == 'RS':
        if activity_type =='MUAe':
            with NixIO(neural_data_path, mode='ro') as reader:
                blk = reader.read_block()
            seg = blk.segments[0]              # there’s only one segment
            anasig = seg.analogsignals[0]      # the MUAe AnalogSignal
            neural_resp = anasig.rescale(pq.uV).magnitude  # now in µV, numpy array
            if z_score is True:
                neural_resp = (neural_resp - neural_resp.mean(axis=0)) / neural_resp.std(axis=0)
            print("Ponce units:", anasig.units)  # should be uV after your rescale
            # print shape just in case
            print(f'neural_resp shape: {neural_resp.shape}')

        if condition_type == 'RS':
            if bin_function is not None:    
                norm_resp = bin_function(neural_resp, window_size=w_size) 
            else:
                norm_resp = neural_resp
        elif condition_type == 'RS_open':
            norm_resp = isolate_RS_resp(neural_resp, date, open_or_closed = 'Open_eyes', monkey=monkey, 
                                            bin_function=bin_function, window_size=w_size)
        elif condition_type == 'RS_closed':
            norm_resp = isolate_RS_resp(neural_resp, date, open_or_closed = 'Closed_eyes', monkey=monkey, 
                                            bin_function=bin_function, window_size=w_size)
        resp_V4, resp_V1 = norm_resp[:,V4_indices], norm_resp[:,V1_indices]
        
        
    # load all neural data matrices
    elif array_condition_type == 'SNR':
        with NixIO(neural_data_path, mode='ro') as reader:
            blk = reader.read_block()
        seg = blk.segments[0]              # there’s only one segment
        anasig = seg.analogsignals[0]      # the MUAe AnalogSignal
        resp_array = anasig.rescale(pq.uV).magnitude  # now in µV, numpy array
        print("Ponce units:", anasig.units)  # should be uV after your rescale
        if z_score is True:
            resp_array = (resp_array - resp_array.mean(axis=0)) / resp_array.std(axis=0)

        true_resp, true_spont = get_epoch_times(resp_array, stim_on=stim_on, stim_off=stim_off, date=date, monkey=monkey, 
                                                sampling_rate=sampling_rate, spont_stim_off=spont_stim_off)
        
        if calculate_SNRs: # this is for the initial SNR calculation. we want to keep all v1, v4, it electrodes with the exception of the artifacts
            return true_resp, true_spont, area_df
        
        if raw_resp:
            if return_binned:
                resp_V4, resp_V1 = true_resp[:,:,V4_indices], true_resp[:,:,V1_indices]
                spont_V4, spont_V1 = true_spont[:,:,V4_indices], true_spont[:,:,V1_indices]
            else:
                resp_V4, resp_V1 = true_resp[:,:,V4_indices].reshape(-1,len(V4_indices)), true_resp[:,:,V1_indices].reshape(-1,len(V1_indices))	
                spont_V4, spont_V1 = true_spont[:,:,V4_indices].reshape(-1,len(V4_indices)), true_spont[:,:,V1_indices].reshape(-1,len(V1_indices))
            if condition_type =='SNR_spont':
                return spont_V4, spont_V1
            elif condition_type =='SNR':
                return resp_V4, resp_V1
        
        
        if bin_function is not None:
            binned_resp = np.stack([bin_function(epoch_resp, w_size, step_size=step_size) for epoch_resp in true_resp])
            binned_spont = np.stack([bin_function(epoch_spont,w_size, step_size=step_size) for epoch_spont in true_spont])
        else:
            binned_resp = true_resp
            binned_spont = true_spont
        
        if subtract_spont_resp:
            norm_resp = binned_resp - np.mean(binned_spont, axis=1, keepdims=True)
        else:
            norm_resp = binned_resp
        
        
        if calculate_relis:
            if return_area_df is True:
                return norm_resp, binned_spont, area_df 
            print('returning binned_resp, binned_spont')
            return norm_resp, binned_spont
        
        # to perform trial shuffling of checkerboard images
        indices = np.arange(len(binned_resp)) 
        # Shuffle the indices using np.random.shuffle
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

        if condition_type =='SNR_spont':
            spont_V4, spont_V1 = binned_spont[:,:,V4_indices], binned_spont[:,:,V1_indices]
            if shuffle is True:
                #just shuffle the responses of V4
                spont_V4 = spont_V4[indices]
                if return_binned:
                    return spont_V4, spont_V1
            spont_V4 = spont_V4.reshape(-1, len(V4_indices))
            spont_V1 = spont_V1.reshape(-1, len(V1_indices))
            return spont_V4, spont_V1
        
        resp_V4, resp_V1 = norm_resp[:,:,V4_indices], norm_resp[:,:,V1_indices]
        diag("PONCE SNR", resp_V4 if resp_V4.ndim==2 else resp_V4.reshape(-1, resp_V4.shape[-1]),
            sampling_rate=1000, window_size_used=w_eff_ponce, subtract_spont_resp=True)
        if shuffle is True:
            #just shuffle the responses of V4
            resp_V4 = resp_V4[indices]
        
        #05oct2025 adding return_binned option with raw_resp False
        if return_binned:
            return resp_V4, resp_V1
        resp_V4 = resp_V4.reshape(-1, len(V4_indices)) # reshape to have shape (n_epochs, n_neurons)
        resp_V1 = resp_V1.reshape(-1, len(V1_indices)) # reshape to have shape (n_epochs, n_neurons)

    return resp_V4, resp_V1


def get_resps_chen(condition_type='SNR', date='090817', monkey='L', w_size = 25, stim_on=0,  stim_off=400, 
            shuffle=False, get_RF_labels=False, bin_function=binning_with_sum, keep_SNR_elecs=False, 
            raw_resp=False, spont_stim_off=300, return_binned=False, SNR_threshold=2, sp_thresh=0.002,
            retrieve_pupil=False, recording_type='MUAe', band=None, return_elec_ids = False, seed=None, 
            plot_psd=False,mains=60, z_score=False, **args):
    """
    This Python function retrieves neural response data based on specified conditions and electrode
    information for further analysis.
    
    :param condition_type: The `condition_type` parameter in the `get_resps` function is used to specify
    the type of condition for which you want to retrieve responses. It can take on different values
    based on the type of data you are interested in analyzing. The function uses this parameter to
    determine which data files to read, defaults to SNR (optional)
    :param date: The `date` parameter in the `get_resps` function is used to specify the date for which
    the data is being retrieved. It is a string parameter that represents the date in the format
    'MMDDYY', defaults to 090817 (optional)
    :param monkey: The `monkey` parameter in the `get_resps` function is used to specify the monkey from
    which the data is being analyzed. It is used to locate the relevant data files and metadata
    associated with the specified monkey for further processing within the function, defaults to L
    (optional)
    :param w_size: The `w_size` parameter in the `get_resps` function represents the window size for
    binning the neural responses. It is used to define the size of the time window over which the neural
    activity will be aggregated or analyzed. This parameter determines the duration of each time bin for
    processing the neural, defaults to 25 (optional)
    :param stim_on: The `stim_on` parameter in the `get_resps` function represents the time point at
    which the stimulus starts in the experiment. It is used to specify the onset time of the stimulus in
    milliseconds relative to the start of the recording, defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `get_resps` function represents the time point at
    which the stimulus ends during data processing. In the provided function, it is set to a default
    value of 400. This means that the stimulus ends at time point 400 in the data processing timeline,
    defaults to 400 (optional)
    :param shuffle: The `shuffle` parameter in the `get_resps` function is a boolean flag that
    determines whether the data should be shuffled before processing. If `shuffle` is set to `True`, the
    data will be randomly shuffled before further processing. If set to `False`, the data will remain in
    its, defaults to False (optional)
    :param get_RF_labels: The `get_RF_labels` parameter in the `get_resps` function is a boolean flag
    that determines whether to return condition labels along with the response arrays for the V4
    cortical area. If `get_RF_labels` is set to `True`, the function will return the response arrays for
    V4, defaults to False (optional)
    :param bin_function: The `bin_function` parameter in the `get_resps` function is used to specify the
    binning function that will be applied to the data. In this case, the default binning function being
    used is `binning_with_sum`. This function likely performs some form of binning operation on the
    :param keep_SNR_elecs: The `keep_SNR_elecs` parameter in the `get_resps` function is a boolean flag
    that determines whether to keep only the electrodes with Signal-to-Noise Ratio (SNR) less than 2,
    defaults to False (optional)
    :param raw_resp: The `raw_resp` parameter in the `get_resps` function is a boolean flag that
    determines whether the raw responses should be returned or not. If `raw_resp` is set to `True`, the
    function will return the raw responses along with any processed data. If `raw_resp` is, defaults to
    False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_resps` function is used to specify
    the time point at which the spontaneous stimulus ends. This parameter is used in the function to
    process neural response data based on the timing of the stimulus presentation and the spontaneous
    activity period. By setting `, defaults to 300 (optional)
    :return: either `resp_V4` and `resp_V1` arrays or `resp_V4`, `resp_V1`, and `cond_labels` arrays
    based on the conditions specified in the function parameters.
    """
    
    data_dir = main_dir + f'data/chen/monkey_{monkey}/{date}/'
    if 'SNR' in condition_type or 'RS' in condition_type: # meaning if its SNR or RS
        SNR_df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_{date}_full.csv')
        if date == '041018':
            SP = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_{date}_removal_metadata.csv')
    
        else:
            SP = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_RS_{date}_removal_metadata.csv')

    elif 'RF' in condition_type: # meaning if its RF
        if monkey == 'L':
            SNR_df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_250717_full.csv')
            SP = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_RS_250717_removal_metadata.csv')
        elif monkey == 'A':
            SNR_df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_041018_full.csv')
            SP = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_041018_removal_metadata.csv') # wanted to remove any electrodes just in case


    if retrieve_pupil:
        pupil_path = os.path.join(main_dir, f'data/chen/monkey_{monkey}/{date}/{monkey}_RS_{date}_pupil_1ms.npz')
        pupil_array = np.load(pupil_path)['pupil']
        # make it 2 dimensional
        if pupil_array.ndim == 1:
            pupil_array = pupil_array[:, np.newaxis]
        binned_pupil = get_clean_array(pupil_array, condition_type, date, monkey,  w_size, stim_on, stim_off, 
                                        bin_function=bin_function, shuffle=shuffle, get_RF_labels=get_RF_labels, raw_resp=raw_resp, 
                                        spont_stim_off=spont_stim_off, return_binned=return_binned, sampling_rate=sampling_rate,
                                        )
    id_dict = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/channel_area_mapping_{monkey}.csv')
    id_snr_filtered = id_dict[id_dict['Electrode_ID'].isin(SNR_df[SNR_df['SNR'] < SNR_threshold]['Electrode_ID'])]
    
    if keep_SNR_elecs is True:
        id_snr_filtered = id_dict[id_dict['Electrode_ID'].isin(SNR_df[SNR_df['SNR'] < -np.inf]['Electrode_ID'])]
    

    id_dict_filtered = id_dict[id_dict['Electrode_ID'].isin(SP['Removed electrode ID'])]
    combined_df = pd.concat([id_dict_filtered, id_snr_filtered])

    SP_SNR_electrodes = {}
    for _, row in combined_df.iterrows():
        array_id = row['Array_ID']
        within_array_electrode_id = row['within_array_electrode_ID']
        SP_SNR_electrodes.setdefault(array_id, set()).add(within_array_electrode_id - 1) #subtracting for python indexing
    uncat_resp_v1 = []
    uncat_resp_v4 = []
    electrode_ids_v1 = []
    electrode_ids_v4 = []
    all_arrays = np.arange(1,17)
    NSPidc = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    if condition_type =='SNR_spont':
        array_condition_type = 'SNR'
    elif 'RS' in condition_type:
        array_condition_type='RS'
    elif 'RF' in condition_type:
        array_condition_type = 'RF'
    else:
        array_condition_type= condition_type
    for a, array in enumerate(all_arrays):
        
        elecs_to_delete = []
        elec_names_to_edit = np.arange(64)
        # Load the data
        with NixIO(data_dir + f'{array_condition_type}/NSP{NSPidc[a]}_array{array}_{recording_type}.nix', mode='ro') as io:
            block = io.read_block()

        # Finding the analog signals
        anasig = block.segments[0].analogsignals
        
        
        if a == 0:
            sampling_rate = anasig[0].sampling_rate.item()
            print(f'Sampling rate: {sampling_rate} Hz')
            window_size = w_size / (1000 / sampling_rate)  # convert w_size from ms to samples
            print("Chen units:", anasig[0].units)  # should be uV
            w_eff_chen  = int(round(w_size / (1000.0 / sampling_rate)))

        del block

        # Find the electrode ID annotations
        within_elec_ID = np.array(anasig[0].array_annotations['within_array_electrode_ID'])
        # Find the electrode ID annotations
        electrode_IDs = np.array(anasig[0].array_annotations['Electrode_ID'])

        # all_annotations = anasig[0].array_annotations
        t_start = int(round(anasig[0].t_start*1000)) # convert to ms
        
        # print(all_annotations.keys())
        array_to_edit = np.array(anasig[0])
        if z_score is True:
            array_to_edit = (array_to_edit - array_to_edit.mean(axis=0)) / array_to_edit.std(axis=0)
        
        cortical_area = anasig[0].array_annotations['cortical_area'][0]
        # print(array_to_edit.shape, cortical_area)
        if date=='140819' and 'RS' in condition_type:
            if a == 0:
                # store t_stop for the first array since this is the true tstop for all arrays
                t_stop = int(round(anasig[0].t_stop*1000))
            if t_start > 0:
                print(f't_start is not 0 for date {date}; will add padding to array {array}')
                padding = np.zeros((t_start, array_to_edit.shape[1]))
                array_to_edit = np.concatenate((padding, array_to_edit), axis=0)
                array_to_edit = array_to_edit[:t_stop, :]
                # print(f'new shape: {array_to_edit.shape}')
        del anasig

        sorted_array_to_edit = array_to_edit[:, np.argsort(within_elec_ID)]
        sorted_electrode_IDs = electrode_IDs[np.argsort(within_elec_ID)]
        elecs_to_delete = np.array([elec for elec in SP_SNR_electrodes.get(array, [])], dtype=int)

        array_to_edit = np.delete(sorted_array_to_edit, elecs_to_delete, axis=1)

        electrode_IDs = np.delete(sorted_electrode_IDs, elecs_to_delete)
        if 'V1' in cortical_area:
            clean_array = get_clean_array(array_to_edit, condition_type, date, monkey, window_size, stim_on, stim_off, 
                                        bin_function=bin_function, shuffle=False, get_RF_labels=False, raw_resp=raw_resp,
                                        spont_stim_off=spont_stim_off, return_binned=return_binned, sampling_rate=sampling_rate,
                                        band=band, plot_psd=plot_psd)
            if clean_array is not None:
                uncat_resp_v1.append(clean_array)
            del array_to_edit
            electrode_ids_v1.extend(electrode_IDs)
        elif 'V4' in cortical_area:
            clean_array = get_clean_array(array_to_edit, condition_type, date, monkey,  window_size, stim_on, stim_off, 
                                        bin_function=bin_function, shuffle=shuffle, get_RF_labels=get_RF_labels, raw_resp=raw_resp, 
                                        spont_stim_off=spont_stim_off, return_binned=return_binned, sampling_rate=sampling_rate,
                                        band=band, plot_psd=plot_psd, mains=mains)
            if clean_array is not None:
                if get_RF_labels is True:
                    uncat_resp_v4.append(clean_array[0])
                    cond_labels = clean_array[1]
                else:
                    uncat_resp_v4.append(clean_array)
            del array_to_edit
            electrode_ids_v4.extend(electrode_IDs)
    if  return_binned is True:
        resp_V1 = np.concatenate(uncat_resp_v1, axis=2)
        resp_V4 = np.concatenate(uncat_resp_v4, axis=2)
        print(f'resp_V4 shape: {resp_V4.shape}, resp_V1 shape: {resp_V1.shape}')
    elif return_binned is True and raw_resp is True:
        resp_V1 = np.concatenate(uncat_resp_v1, axis=2)
        resp_V4 = np.concatenate(uncat_resp_v4, axis=2)
    else:
        resp_V1 = np.concatenate(uncat_resp_v1, axis=1)
        resp_V4 = np.concatenate(uncat_resp_v4, axis=1)
    diag("CHEN SNR",  resp_V4 if resp_V4.ndim==2 else resp_V4.reshape(-1, resp_V4.shape[-1]),
        sampling_rate=sampling_rate, window_size_used=w_eff_chen, subtract_spont_resp=True)
    if return_elec_ids:
        if get_RF_labels is True:
            return resp_V4, resp_V1, cond_labels, np.array(electrode_ids_v4), np.array(electrode_ids_v1)
        if retrieve_pupil:
            return resp_V4, resp_V1, binned_pupil, np.array(electrode_ids_v4), np.array(electrode_ids_v1)
        return resp_V4, resp_V1, np.array(electrode_ids_v4), np.array(electrode_ids_v1)
    if get_RF_labels is True:
        return resp_V4, resp_V1, cond_labels
    if retrieve_pupil:
        return resp_V4, resp_V1, binned_pupil
    return resp_V4, resp_V1


def get_clean_array(resp_array, condition_type='SNR', date='090817', monkey='L', w_size = 25, stim_on=0, 
                    stim_off=400, bin_function=binning_with_sum, shuffle=False, get_RF_labels=False, raw_resp=False, 
                    spont_stim_off=200, return_binned=False, sampling_rate=1000, band=None, plot_psd=False,
                    mains=60):
    """
    This Python function takes in response data and parameters to clean and process the data based on
    different conditions such as signal-to-noise ratio (SNR), resting state (RS), and receptive field
    (RF).
    
    :param resp_array: `resp_array` is the input array containing response data. The function
    `get_clean_array` processes this array based on the specified conditions and parameters to return a
    cleaned and processed version of the data. The function handles different types of conditions such
    as SNR, RS, RF, and their variations,
    :param condition_type: The `condition_type` parameter in the `get_clean_array` function determines
    the type of data processing to be applied to the input `resp_array`. The function supports different
    condition types for processing the data, such as 'SNR', 'SNR_spont', 'RS', 'RS_open',, defaults to
    SNR (optional)
    :param date: The `date` parameter in the `get_clean_array` function is used to specify the date for
    data processing. It is a string parameter that represents the date in a specific format, such as
    '090817' in the function call, defaults to 090817 (optional)
    :param monkey: The `monkey` parameter in the `get_clean_array` function is used to specify the
    monkey from which the response data is collected. It is a string parameter that can take values like
    'L' or 'R' to indicate the left or right monkey, for example, defaults to L (optional)
    :param w_size: The `w_size` parameter in the `get_clean_array` function represents the window size
    used for binning the response data. It is an integer value that determines the size of the window
    for aggregating or summarizing the responses within each window, defaults to 25 (optional)
    :param stim_on: The `stim_on` parameter in the `get_clean_array` function represents the time point
    when the stimulus starts in the experiment. It is used in the calculation of responses based on
    different conditions such as SNR (Signal-to-Noise Ratio) or RF (Receptive Field). The value of `,
    defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `get_clean_array` function represents the time
    point at which the stimulus ends during data processing. It is used in various conditions like 'SNR'
    and 'RF' to determine the duration of the stimulus presentation, defaults to 400 (optional)
    :param bin_function: The `bin_function` parameter in the `get_clean_array` function is used to
    specify the function that will be applied to bin the response array data. This function will take
    the response array and window size as input parameters and return the binned data. The function can
    be customized based on the specific
    :param shuffle: The `shuffle` parameter in the `get_clean_array` function is a boolean flag that
    determines whether the data should be shuffled or not. If `shuffle` is set to `True`, the data will
    be shuffled before processing. If `shuffle` is set to `False`, the data will not, defaults to False
    (optional)
    :param get_RF_labels: The `get_RF_labels` parameter is a boolean flag that determines whether the
    function should return both the `sum_binned` array and the `binned_labels` array when set to `True`.
    If `get_RF_labels` is `True`, the function will return both arrays; otherwise, it, defaults to False
    (optional)
    :param raw_resp: The `raw_resp` parameter in the `get_clean_array` function is a boolean flag that
    specifies whether to use raw response data or not. If `raw_resp` is set to `True`, the function will
    use raw response data in the calculations. If `raw_resp` is set to `, defaults to False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_clean_array` function is used to
    specify the time point at which the spontaneous stimulus ends. This parameter is used in the
    'SNR_spont' and 'RF_spont' conditions to determine the duration of the spontaneous stimulus period,
    defaults to 200 (optional)
    :return: either `sum_binned` or a tuple containing `sum_binned` and `binned_labels` based on the
    conditions specified in the function.
    """
    if band is not None:
        if band not in BANDS:
            raise ValueError(f"Band '{band}' not recognized. Available bands: {list(BANDS.keys())}")
        # f1, f2 = BANDS[band]
        # print(f'Filtering LFP to {band} band ({f1}-{f2} Hz)')
        # print(f'resp_array shape before filtering: {resp_array.shape}')
        resp_array = lfp_to_band_envelopes(resp_array, fs=sampling_rate, mains=mains, notch_Q=35, 
                                        log_amp=False, zscore=False, band=band, plot_psd=plot_psd)
        # print(f'Filtered LFP to {band} band ({f1}-{f2} Hz)')
        # print(f'resp_array shape after filtering: {resp_array.shape}')
    
    if condition_type == 'SNR_spont':
        sum_binned = isolate_norm_spont(resp_array=resp_array, bin_function=bin_function, window_size=w_size, date=date, 
                                        monkey=monkey, shuffle=shuffle, raw_resp=raw_resp, spont_stim_off=spont_stim_off,
                                        return_binned=return_binned, sampling_rate=sampling_rate)
    elif condition_type == 'SNR':
        sum_binned = isolate_norm_resps(resp_array, stim_on=stim_on, 
                                        stim_off=stim_off, 
                                        bin_function=bin_function, 
                                        window_size=w_size, date=date, 
                                        monkey=monkey, shuffle=shuffle, raw_resp=raw_resp,
                                        return_binned=return_binned,
                                        sampling_rate=sampling_rate) 
    elif condition_type == 'RS':
        if bin_function is not None:
            sum_binned = bin_function(resp_array, window_size=w_size)
            
        else:
            sum_binned = resp_array
        # sum_binned -= np.mean(sum_binned,axis=0)
        
    elif condition_type == 'RS_open':
        sum_binned = isolate_RS_resp(resp_array, date, open_or_closed = 'Open_eyes', monkey=monkey, 
                                        bin_function=bin_function, window_size=w_size, sampling_rate=sampling_rate)
    elif condition_type == 'RS_closed':
        sum_binned = isolate_RS_resp(resp_array, date, open_or_closed = 'Closed_eyes', monkey=monkey, 
                                        bin_function=bin_function, window_size=w_size, sampling_rate=sampling_rate)

    elif condition_type == 'RF':
        sum_binned, binned_labels = isolate_norm_resps_RF(resp_array, stim_on=stim_on, stim_off=stim_off,
                                                        bin_function=bin_function, window_size=w_size,
                                                        date=date, monkey=monkey, raw_resp=raw_resp,
                                                        return_binned=return_binned, sampling_rate=sampling_rate)
    
    elif condition_type == 'RF_spont':
        sum_binned, binned_labels = isolate_norm_spont_RF(resp_array, stim_on=stim_on, stim_off=stim_off,
                                                        bin_function=bin_function, window_size=w_size,
                                                        date=date, monkey=monkey, raw_resp=raw_resp,
                                                        spont_stim_off=spont_stim_off, return_binned=return_binned,
                                                        sampling_rate=sampling_rate)

    del resp_array

    if get_RF_labels is True:
        return sum_binned, binned_labels

    return sum_binned

def get_get_condition_type(condition_type):
    """
    The function `get_get_condition_type` returns a modified condition type based on specific criteria.
    
    :param condition_type: The function `get_get_condition_type` takes a `condition_type` as input and
    returns a modified version of it based on certain conditions
    :return: The function `get_get_condition_type` returns the value of the variable
    `get_condition_type`, which is determined based on the input `condition_type`. If the input contains
    both 'RF' and 'spont', it returns 'RF_spont'. If the input contains only 'RF', it returns 'RF'.
    Otherwise, it returns the original `condition_type`.
    """
    if 'RF' in condition_type and 'spont' in condition_type:
        get_condition_type='RF_spont'
    elif 'RF' in condition_type:
        get_condition_type='RF'
    else:
        get_condition_type=condition_type
    return get_condition_type


import numpy as np
from scipy.signal import iirnotch, tf2sos, sosfiltfilt

def notch_once(X, fs, f0, Q=35):
    if f0 >= fs/2: return X
    b, a = iirnotch(f0/(fs/2), Q)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, X, axis=0)

def notch_mains_and_harmonics(lfp_txch, fs=500, mains=50, harmonics=(1,2,3), Q=35):
    
    X = lfp_txch
    for h in harmonics:
        f0 = h*mains
        if f0 < fs/2:
            X = notch_once(X, fs, f0, Q=Q)
    return X

from scipy.signal import iirfilter, hilbert, sosfiltfilt

# Morales-Gregorio/Chen bands
BANDS = {
    "low_2_12":   (2, 12),
    "beta_12_30": (12, 30),
    "gamma_30_45":(30, 45),
    "hgamma_55_95":(55, 95),
}

def bandpass_sos(fs, f1, f2, order=4):
    return iirfilter(order, [f1/(fs/2), f2/(fs/2)],
                     btype='band', ftype='butter', output='sos')

def lfp_to_band_envelopes(lfp_txch, fs=500, mains=60, notch_Q=35, log_amp=False, zscore=False, 
                          band='beta_12_30',plot_psd=False, psd_fmax=200, psd_chan_idx=None, psd_title=None,
                          adaptive_notch=True):
    """
    lfp_txch: (T, C) continuous LFP (one array at a time)
    returns envelope (T, C) for the requested band.
    If plot_psd=True, shows median PSD (channels in psd_chan_idx) pre- vs post-notch.
    adaptive_notch: if True, estimates local peak near mains and harmonics before notching.
    """
    X_pre = lfp_txch.astype(np.float64, copy=False)

    if adaptive_notch:
        # snap each target to the actual local peak (works for “58”, “119.8”, “125”, etc.)
        targets = [mains, 2*mains, 3*mains]  # tweak per dataset
        freqs = [estimate_peak_near(X_pre, fs, f0, search_hz=3) for f0 in targets]
        X = notch_at_frequencies(X_pre, fs, freqs, Q=notch_Q)
    else:
        # fixed notches (still great in practice)
        X = notch_at_frequencies(X_pre, fs, [mains, 2*mains, 3*mains], Q=notch_Q)
        

    if plot_psd:
        ttl = psd_title or f"PSD pre/post notch — fs={fs} Hz, mains={mains}"
        _plot_psd_compare_pre_post(X_pre, X, fs, fmax=psd_fmax, chan_idx=psd_chan_idx, title=ttl)
    # 2) band → Hilbert amplitude
    f1, f2 = BANDS[band]
    
    sos = bandpass_sos(fs, f1, f2, order=4)
    xb  = sosfiltfilt(sos, X, axis=0)             # zero-phase
    amp = np.abs(hilbert(xb, axis=0))             # envelope
    if log_amp:
        amp = np.log(amp + 1e-12)
    if zscore:
        m = amp.mean(axis=0, keepdims=True)
        s = amp.std(axis=0, keepdims=True) + 1e-12
        amp = (amp - m)/s
    return amp

def events_to_indices(event_times_s, fs):
    return np.asarray([int(round(t*fs)) for t in event_times_s], dtype=int)

def epoch_array(X_txch, idxs, fs, t_pre_s, t_post_s):
    pre  = int(round(t_pre_s  * fs))
    post = int(round(t_post_s * fs))
    T, C = X_txch.shape
    trials = []
    kept = []
    for i, idx in enumerate(idxs):
        a, b = idx - pre, idx + post
        if a >= 0 and b <= T:
            trials.append(X_txch[a:b])
            kept.append(i)
    return np.stack(trials, axis=0), np.array(kept)   # (n_trials, T_win, C)

########################################## PONCE LFP RELEVANT FUNCTION S ##########################################
import numpy as np
from scipy.signal import iirfilter, sosfiltfilt, hilbert, iirnotch, tf2sos, resample_poly


import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt


# ---- small PSD helper (median across channels) ----
def _median_psd(X_txch, fs, chan_idx=None, nperseg=4096):
    if chan_idx is None:
        chan_idx = range(X_txch.shape[1])
    pxxs = []
    for ch in chan_idx:
        f, pxx = welch(X_txch[:, ch], fs=fs, nperseg=nperseg)
        pxxs.append(pxx)
    pxxs = np.vstack(pxxs)
    return f, np.nanmedian(pxxs, axis=0)

def _plot_psd_compare_pre_post(X_pre, X_post, fs, fmax=200, chan_idx=None, title=None):
    f, p_pre  = _median_psd(X_pre,  fs, chan_idx=chan_idx)
    _, p_post = _median_psd(X_post, fs, chan_idx=chan_idx)
    m = f <= fmax
    plt.figure(figsize=(5,3))
    plt.semilogy(f[m], p_pre[m], label="pre-notch", alpha=0.8)
    plt.semilogy(f[m], p_post[m], label="post-notch", alpha=0.8)
    # mark common line freqs for both EU and US
    for v in (50, 60, 100, 120, 150):
        if v < fs/2 and v <= fmax:
            plt.axvline(v, ls="--", lw=1, alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    if title: plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
    
from scipy.signal import welch, iirnotch, tf2sos, sosfiltfilt
import numpy as np

def estimate_peak_near(X_txch, fs, f0, search_hz=3, nperseg=4096):
    """Find the frequency of the largest PSD peak within ±search_hz of f0 (median across channels)."""
    # median PSD across channels for robustness
    pxxs = []
    for ch in range(X_txch.shape[1]):
        f, p = welch(X_txch[:, ch], fs=fs, nperseg=nperseg)
        pxxs.append(p)
    pxx = np.nanmedian(np.vstack(pxxs), axis=0)
    m = (f >= f0 - search_hz) & (f <= f0 + search_hz)
    if not m.any():
        return float(f0)
    return float(f[m][np.argmax(pxx[m])])

def notch_at_frequencies(X_txch, fs, freqs, Q=35):
    """Notch an explicit list of frequencies (floats allowed)."""
    Y, nyq = X_txch, fs/2.0
    for f0 in np.atleast_1d(freqs):
        if 0 < f0 < nyq:
            b, a = iirnotch(f0/(fs/2), Q)
            Y = sosfiltfilt(tf2sos(b, a), Y, axis=0)
    return Y
import numpy as np
import quantities as pq

def diag(name, x_txch, sampling_rate, window_size_used, subtract_spont_resp=True):
    print(f"\n=== {name} ===")
    print("shape (T×C):", x_txch.shape)
    print("fs (Hz):", sampling_rate)
    print("window_size (samples):", window_size_used)
    print("subtract_spont_resp:", subtract_spont_resp)
    # pre-binning/raw scale (if you still have it around)
    # print("raw µV p5/50/95:", np.percentile(raw_uV, [5,50,95]))
    print("post-binning µV p5/50/95:", np.percentile(x_txch, [5,50,95]))
    print("per-channel median µV (median across channels):",
          np.median(np.median(x_txch, axis=0)))