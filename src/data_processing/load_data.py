import pandas as pd
import numpy as np
import os
import time


def get_sleep(birthday, DATA_PATH):
    '''
    This function loads raw .csv data acquired from the 'BabyTracker' app and returns
    a binary time series describing a sleep pattern where '1' corresponds to 'sleep' state and '0' - to 'awake'.

    The output contains the sleep pattern over the last two months where the timestamps are shifted
    such that the infants' birthday is effectively set to 01/01/1900 at 0:00 am.
    The function also returns a timestamp starting from which we will make a prediction.
    '''

    data = pd.read_csv(os.path.join(DATA_PATH, 'raw/csv/Alisa_sleep.csv'), parse_dates=['Time'], index_col='Time')
    data.columns = ['baby', 'duration', 'note']
    data = data.drop(['baby', 'note'], axis=1)
    data = data[data.duration >= 15]

    # get the time after which we would like to make a prediction
    modtimesec = os.path.getmtime(os.path.join(DATA_PATH, 'raw/csv/Alisa_sleep.csv'))
    finaltime = pd.to_datetime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modtimesec)))

    # calculate the number of minutes between the
    finaldelta = (finaltime - data.index[0]) / np.timedelta64(1, 'm')
    finaltime = finaltime.round(freq='10min')

    # offset the overall date
    anchordate = pd.to_datetime('01/01/2000')
    subdate = pd.to_datetime(birthday) - anchordate
    data.index -= subdate

    # pick out data over the last 2 months and 5 days
    time_cut = data.index[0] - pd.DateOffset(months=2, days=5)
    data = data[data.index > time_cut]

    time_start = data.index[-1]
    data['minutes'] = (data.index - time_start) / np.timedelta64(1, 'm')
    data.minutes = data.minutes.apply(int)

    # generate a binary timeseries of sleep/awake status with 10-minute steps
    div = 10
    maxt = data['minutes'][0] // div
    data_bin = np.zeros(maxt + int(finaldelta // div), dtype=int)
    interval = -1
    for i in range(maxt):
        if data.minutes[interval] <= div * i <= data.minutes[interval] + data.duration[interval]:
            data_bin[i] = 1
        elif div * i > data.minutes[interval] + data.duration[interval]:
            while div * i > data.minutes[interval] + data.duration[interval]:
                interval -= 1
    dsf = pd.DataFrame(data_bin, dtype=int)
    dsf.columns = ['sleep']
    dsf = dsf[-60 * 144:]
    dsf.index = pd.date_range(end=(finaltime - subdate).round(freq='10min'), periods=60 * 144, freq='10min')

    #load NYU processed data to add additional predictive features needed for our models
    trend = pd.read_csv(os.path.join(DATA_PATH, 'NYU_data_processed/NYU_trend.csv')
                        , parse_dates=['timestamps'], index_col= 'timestamps')

    # additional features
    dsf['trend'] = trend.trend
    dsf['time'] = trend.time
    dsf['time_cos'] = trend.time_cos
    dsf['ampm'] = trend.ampm

    return dsf, finaltime
