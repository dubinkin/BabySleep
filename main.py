import src.data_processing.load_data as loaddata
import src.models.LGBM_next_step as LGBM1step
import matplotlib.pyplot as plt

sleep, time0 = loaddata.get_sleep('12/06/2021', 'data/')

prediction = LGBM1step.forecast(sleep, time0, 'data/')

fig, ax = plt.subplots(figsize=(16, 4))
prediction.plot(ax = ax)
fig.savefig('prediction.png')