import pandas as pd
import lightgbm as lgb

def prepare_data(sleep, window):
    '''
    Prepares data in a format accessible by the model.

    :param sleep: sleep data generated by the load_data.get_sleep function
    :param window: number of 10-minute steps which are included in a given
                       example which are used to make a forecast
    :return: training examples with labels and the data which is used to
             start the next-step forecasting
    '''

    infant = sleep.sleep
    n = infant.shape[0]
    pred_window = 10

    X_train = pd.DataFrame([infant.values[i : i + window]
                            for i in range(n - window)])
    y_train = pd.DataFrame([sum(infant.values[i + window :
                                              i + window + pred_window])
                            for i in range(n - window - pred_window)])

    newsleep = sleep[window:].reset_index(drop=True)
    X_train['time_of_day'] = newsleep.time
    X_train['cos'] = newsleep.time_cos
    X_train['AM/PM'] = newsleep.ampm

    for j in range(6, 6*24+1, 6):
        X_train[str(j) + '_10min'] = pd.DataFrame(
            [infant.values[i + window - j: i + window].sum()
             for i in range(n - window)])

    X_train['N_naps_12hrs'] = pd.DataFrame(
        [abs(pd.Series(infant.values[i + window - 72 :
                                     i + window]).diff()).sum()//2
         for i in range(n - window)])

    return X_train[ : -pred_window], y_train, X_train[-pred_window :]


def train_model(X_train, y_train):
    '''
    Function which fits the model on the training data. The optimal model
    parameters were found previously via a simple grid search.

    :param X_train: training examples
    :param y_train: training labels
    :return: trained lgbm regressor
    '''

    lgbm_reg = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=25,
                                 max_depth=-1, learning_rate=0.1,
                                 n_estimators=300, subsample_for_bin=200000,
                                 objective=None, class_weight=None,
                                 min_split_gain=0.0, min_child_weight=0.001,
                                 min_child_samples=20, subsample=1.0,
                                 subsample_freq=0, colsample_bytree=1.0,
                                 reg_alpha=0.0, reg_lambda=0.0,
                                 random_state=42, n_jobs = -1,
                                 importance_type='split')

    lgbm_reg.fit(X_train, y_train.values.reshape(-1))

    return lgbm_reg

def sleepy_score(sleep):
    '''
    Generate a score telling how many of the next 10-minute intervals the model
    predicts will be in a 'asleep' state.

    :param sleep: sleep data generated by the load_data.get_sleep function
    :return: a real number between 0 and 10 corresponding to the number of
             'asleep' intervals among the next 10 time steps
    '''

    window = 6 * 24 * 6
    X_train, y_train, Xtt = prepare_data(sleep, window)
    model = train_model(X_train, y_train)

    score = model.predict(Xtt[-1 :])
    if score > 10:
        score = 10
    if score < 0:
        score = 0

    return score