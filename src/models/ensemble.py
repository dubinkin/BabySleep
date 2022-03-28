import src.data_processing.prediction_processing as predprocess
import src.models.LGBM_next_step as LGBM1step
import src.models.LGBM_avg as LGBMavg
import src.models.SARIMAX as SARIMAXfc
import numpy as np
import pandas as pd


def forecast(sleep, time0, DATA_PATH, MODEL_PATH,
             forecast_period, weights, comb_conv, output_conv):
    '''
    Function which combines the predictions of all models in the ensemble and
    returns a combined, smoothed-out sleep/awake prediction as well as the
    predictions generated by all individual models.

    :param sleep: sleep data generated by the load_data.get_sleep function
    :param time0: timestamp after which the forecast is being made
    :param DATA_PATH: path to the data folder containing NYU dataset
    :param MODEL_PATH: path to the folder with ML models
    :param forecast_period: number of steps to make a prediction for
    :param weights: weights with which the individual models are combined
    :param comb_conv: filter which smooths out individual predictions
                      before combining them
    :param output_conv: filter which smooths out final prediction
    :return: combined prediction and individual predictions for each model
    '''

    # check that weights are properly normalized
    weights = weights / weights.sum()

    fc1 = LGBM1step.forecast(sleep, time0, DATA_PATH, forecast_period)
    fc2 = LGBMavg.forecast(sleep, time0, forecast_period)
    fc3 = SARIMAXfc.forecast(sleep, time0, DATA_PATH,
                             MODEL_PATH, forecast_period)

    # generate smoother version of these predictions appropriate for merging
    startpt = 64
    sm_fc1 = predprocess.smooth(fc1, comb_conv, startpt)
    sm_fc2 = predprocess.smooth(fc2, comb_conv, startpt)
    sm_fc3 = predprocess.smooth(fc3, comb_conv, startpt)

    fcc = weights[0]*sm_fc1 + weights[1]*sm_fc2 + weights[2]*sm_fc3

    # check that the predicted amount of sleep is in the well-expected
    # range based on the data for previous similar periods
    daysleeps = np.array([])
    for i in range(50):
        sleepval = sleep.sleep[- 1 - (i+1) * 144 : - 1 - i * 144].values.sum()
        daysleeps = np.append(daysleeps, sleepval)

    sleepup = daysleeps.mean() + 2*daysleeps.std()
    sleepdw = daysleeps.mean() - 2*daysleeps.std()

    threshold = 0.5
    fcc_bin = pd.DataFrame(index=fcc.index, data=fcc.trend)
    fcc_bin.trend = (fcc.trend > threshold).astype(int)
    totalsleep = fcc_bin[64 : 64 + 144].values.sum()

    if totalsleep > sleepup:
        while (totalsleep > sleepup) and (threshold > 0.1):
            threshold -= 0.05
            fcc_bin.trend = (fcc.trend > threshold).astype(int)
            totalsleep = fcc_bin[64 : 64 + 144].values.sum()

    elif totalsleep < sleepdw:
        while (totalsleep < sleepdw) and (threshold < 0.9):
            threshold += 0.05
            fcc_bin.trend = (fcc.trend > threshold).astype(int)
            totalsleep = fcc_bin[64 : 64 + 144].values.sum()

    fcc_final = predprocess.smooth(fcc_bin, output_conv,
                                   startpt)[58 : 64 + 144]

    return fcc_final, sm_fc1, sm_fc2, sm_fc3