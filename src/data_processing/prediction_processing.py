import numpy as np
import matplotlib.pyplot as plt
import os


def smooth(prediction, conv, start):
    '''
    Returns smoothed out prediction time series, by applying a convolution filter
    'conv' to a raw binary prediction series 'prediction' starting from the point indicated by 'start'.

    :param prediction: binary timeseries representing raw sleep prediction
    :param conv: convolution filter used for smoothing the raw series out
    :param start: an index of the prediction array after which the series will be smoothed out
    :return: sm_prediction - smoother out prediction series
    '''

    conv = conv / conv.sum()

    if conv.shape[0] % 2 == 0:
        print('Convoluton filter has even length and cannot be centered!')
        return prediction

    n = prediction.shape[0]
    k = conv.shape[0] // 2
    predvals = prediction.trend.tolist()
    sm_prediction = prediction.copy()

    for i in range(n - start - 2 * k):
        sm_prediction.iloc[start + 1 + i] = np.dot(predvals[start + 1 + i - k : start + 1 + i + k + 1], conv)

    return sm_prediction[: - 2 * k]

def plot_predict(ans, PREDICT_PATH, NAME):
    fig, ax = plt.subplots(figsize = (10, 4))
    ans.plot(ax = ax)
    plt.fill_between(x = ans.index, y1 = ans.trend, alpha = 0.25)
    plt.yticks([])
    plt.text(-0.05,0.95,'Sleeping',horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes)
    plt.text(-0.05,0.05,'Awake',horizontalalignment='center',
             verticalalignment='center', transform = ax.transAxes)
    ax.get_legend().remove()
    fig.savefig(os.path.join(PREDICT_PATH, NAME))

    return None