import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import src.data_processing.load_data


def prepare_data(sleep, window):
    infant = sleep.sleep
    n = infant.shape[0]
    pred_window = 6

    X_train = pd.DataFrame([infant.values[i : i + window]
                            for i in range(n - window)])
    y_train = pd.DataFrame([sum(infant.values[i + window : i + window + pred_window])
                            for i in range(n - window - pred_window)])

    newsleep = sleep[window:].reset_index(drop = True)
    X_train['time_of_day'] = newsleep.time
    X_train['cos'] = newsleep.time_cos
    X_train['AM/PM'] = newsleep.ampm

    for j in range(6, 6*24+1, 6):
        X_train[str(j) + '_10min'] = pd.DataFrame([infant.values[i + window - j: i + window].sum() for i in range(n - window)])

    X_train['N_naps_12hrs'] = pd.DataFrame([abs(pd.Series(infant.values[i + window - 72: i + window]).diff()).sum()//2
                                            for i in range(n - window)])

    return X_train[ : -pred_window], y_train, X_train[-pred_window :]


def train_model(X_train, y_train):
    lgbm_reg = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=20, max_depth=-1, learning_rate=0.1,
                              n_estimators=250, subsample_for_bin=200000, objective=None,
                              class_weight=None, min_split_gain=0.0, min_child_weight=0.001,
                              min_child_samples=20, subsample=1.0, subsample_freq=0,
                              colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=42,
                              n_jobs = -1, importance_type='split')

    lgbm_reg.fit(X_train, y_train.values.reshape(-1))

    return lgbm_reg


def prediction(X, model, weights, window, rounding = True):

    if X.shape[0] != 6:
        print("Wrong input data!")
        return -1
    weights = weights / np.sum(weights)

    last_status = X.iloc[-1][window - 1]

    if last_status == 1:
        vec = ((model.predict(X).round() - [5, 4, 3, 2, 1, 0]) > 0).astype(int)
        if rounding:
            return round(np.dot(vec, weights))
        else:
            return np.dot(vec, weights)
    else:
        vec = ((model.predict(X).round() - [5, 4, 3, 2, 1, 0]) < 1).astype(int)
        if rounding:
            return round(1 - np.dot(vec, weights))
        else:
            return 1 - np.dot(vec, weights)


def data_augment(X, prediction, time, window):
    X_c = X.iloc[:, : window]
    X_c = X_c.shift(-1, axis = 1)
    X_c.iloc[0, window-1] = prediction
    X_c.index += 1
    time = time + 24/144
    X_c['time_of_day'] = time
    X_c['cos'] = np.cos(2*np.pi * time / 24)
    if time > 12:
        X_c['AM/PM'] = 1
    else:
        X_c['AM/PM'] = 0
    for j in range(6,6*24+1,6):
        X_c[str(j) + '_10min'] = X_c.values[:, window - j : window].sum()
    X_c = X_c.astype(int)
    X_c['N_naps_12hrs'] = pd.Series(X_c.values[:, window - 72: window].flatten()).diff().abs().sum()//2

    return X_c


def update_step(X, model, weights, window):
    pred = prediction(X, model, weights, window)
    X_c = X[1:]
    new_row = data_augment(X[5:], pred, X[5:].time_of_day.values, window)
    X_c = pd.concat([X_c,new_row])

    return X_c


def lgbm_avg_prediction(X, model, weights, window, forecast_period):
    sleep = np.array(X.iloc[-1, window - 64 : window])

    for i in range(forecast_period):
        X = update_step(X, model, weights, window)
        sleep = np.append(sleep, X.iloc[-1, window - 1])

    return sleep

def forecast(sleep, time0, forecast_period):
    window = 6 * 24 * 6
    X_train, y_train, Xtt = prepare_data(sleep, window)
    model = train_model(X_train, y_train)

    weights = np.array([1, 1, 1, 1, 1.1, 1.25])

    futuresleep = lgbm_avg_prediction(Xtt, model, weights, window, forecast_period)

    times = pd.date_range(start = sleep.index[-65], periods = 64 + forecast_period, freq = '10min')
    answer = pd.DataFrame(data = futuresleep, index = times, columns = ['trend'])

    answer.index += time0 - answer.index[64]

    return answer



def predict_proba(sleep):
    window = 6 * 24 * 6
    X_train, y_train, Xtt = prepare_data(sleep, window)
    model = train_model(X_train, y_train)

    weights = np.array([1, 1, 1, 1, 1.1, 1.25])

    predval = prediction(Xtt, model, weights, window, rounding = False)

    if predval > 1:
        predval = 1
    if predval < 0:
        predval = 0

    return predval

