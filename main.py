import src.data_processing.load_data as load_data
import src.data_processing.prediction_processing as predprocess
import src.models.LGBM_sleepy_score as sscore
import src.models.ensemble as ensemble
import src.models.SARIMAX as SARIMAXfc
import numpy as np

#load baby's data
sleep, time0 = load_data.get_sleep('12/06/2021', 'data/')

#generate sleepy score prediction
score = sscore.sleepy_score(sleep)
print('Raw sleepy score:', score)
if score < 2.5:
    score = 0
elif score < 5:
    score = 1
elif score <= 10:
    score = 2
sleepdict = ['Not sleepy at all', 'Somewhat sleepy', 'Very sleepy']
print('Sleepy score: %1d out of 3' % (score+1))
print(sleepdict[score])

#define smoothing function which helps to average over several predictions
weights = np.array([1, 1, 0])
conv = np.array([1, 1, 2, 4, 6, 4, 2, 1, 1])
conv = conv / conv.sum()

#define needed variables
forecast_period = 1 * 24 * 6 + 30
DATA_PATH = 'data/'
MODEL_PATH = 'src/models/'

#compute the
ans, pred1, pred2, pred3 = ensemble.forecast(sleep, time0, DATA_PATH, MODEL_PATH, forecast_period, weights, conv, conv)

#generate plots of the main forecast, as well as the plots for individual models' predictions
predprocess.plot_predict(ans, 'predictions/', 'forecast.png')
predprocess.plot_predict(pred1, 'predictions/', 'prediction_LGBM_1step.png')
predprocess.plot_predict(pred2, 'predictions/', 'prediction_LGBM_avg.png')
#predprocess.plot_predict(pred3, 'predictions/', 'prediction_SARIMAX.png')


#update the model if needed
SARIMAXfc.model_update(sleep, time0, MODEL_PATH)


