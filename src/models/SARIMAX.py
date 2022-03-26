import pandas as pd
import os
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

def forecast(sleep, time0, DATA_PATH, MODEL_PATH, forecast_period):

    timeseries = sleep.sleep - sleep.trend

    #load the latest fitted model
    res = SARIMAXResults.load(os.path.join(MODEL_PATH, 'saved_models/SARIMAX.pkl'))

    #update model with the new data
    file = open(os.path.join(MODEL_PATH, 'saved_models/SARIMAX_time.txt'), 'r')
    last_time = pd.Timestamp(file.read())
    file.close()
    timeseries0 = timeseries[timeseries.index > last_time]
    res = res.append(timeseries0)
    res.save(os.path.join(MODEL_PATH, 'saved_models/SARIMAX.pkl'))
    file = open(os.path.join(MODEL_PATH, 'saved_models/SARIMAX_last_update.txt'), 'w')
    file.write(time0.strftime('%Y-%m-%d %X'))
    file.close()

    #get the forecast
    forecast = res.get_forecast(steps = forecast_period)
    prediction = pd.Series(forecast.predicted_mean)
    answer = pd.concat([timeseries[-64:],prediction])
    answer = pd.DataFrame(answer, columns=['trend'])

    trend = pd.read_csv(os.path.join(DATA_PATH, 'NYU_data_processed/NYU_trend.csv'), parse_dates=['timestamps'], index_col= 'timestamps')
    addtrend = pd.DataFrame(trend.trend[answer.index[0]:answer.index[-1]])

    answer = answer + addtrend
    answer.trend = (answer.trend > 0.5).astype(int)
    answer.index += time0 - answer.index[63]

    return answer

def model_update(sleep, time0, MODEL_PATH):

    #get the date when the SARIMAX model was last updated
    file = open(os.path.join(MODEL_PATH, 'saved_models/SARIMAX_last_update.txt'), 'r')
    model_last_update = pd.Timestamp(file.read())
    file.close()

    #update the model if it's more than 5 days old
    if time0 - model_last_update > pd.Timedelta(days = 3):
        timeseries = (sleep.sleep - sleep.trend)[-10 * 24 * 6 - 12 * 6 :]
        mod = SARIMAX(timeseries,
                      order=(10, 0, 0),
                      seasonal_order = (2, 1, 1, 144))
        res = mod.fit()
        res.save(os.path.join(MODEL_PATH, 'saved_models/SARIMAX.pkl'))

        file = open(os.path.join(MODEL_PATH,'saved_models/SARIMAX_time.txt'), 'w')
        file.write(timeseries.index[-1].strftime('%Y-%m-%d %X'))
        file.close()
        file = open(os.path.join(MODEL_PATH, 'saved_models/SARIMAX_last_update.txt'), 'w')
        file.write(time0.strftime('%Y-%m-%d %X'))
        file.close()


def predict_proba(sleep, DATA_PATH, MODEL_PATH):

    timeseries = sleep.sleep - sleep.trend

    #load the latest fitted model
    res = SARIMAXResults.load(os.path.join(MODEL_PATH, 'saved_models/SARIMAX.pkl'))

    #update model with the new data
    file = open(os.path.join(MODEL_PATH, 'saved_models/SARIMAX_time.txt'), 'r')
    last_time = pd.Timestamp(file.read())
    timeseries0 = timeseries[timeseries.index > last_time]
    res = res.append(timeseries0)

    #get the forecast
    forecast = res.get_forecast(steps = 1)
    prediction = pd.Series(forecast.predicted_mean)
    answer = pd.concat([timeseries[-64:],prediction])
    answer = pd.DataFrame(answer, columns=['trend'])

    trend = pd.read_csv(os.path.join(DATA_PATH, 'NYU_data_processed/NYU_trend.csv'), parse_dates=['timestamps'], index_col= 'timestamps')
    addtrend = pd.DataFrame(trend.trend[answer.index[0]:answer.index[-1]])

    answer = answer + addtrend
    predval = answer.iloc[-1, 0]
    if predval > 1:
        predval = 1
    if predval < 0:
        predval = 0

    return predval