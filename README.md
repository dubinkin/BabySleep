# BabySleep
An ensemble of ML models which examines your baby's previous sleep patterns and uses this data to tell you how sleepy your baby is right now and to suggest most natural sleeping windows in the next 24 hours so that you can better plan your day.

It comes together with a telegram bot allowing to access this model on the go!
Here's a short demonstration of how this all works in practice:

https://user-images.githubusercontent.com/17629106/160226505-2502b6f6-e811-4d28-8b32-77d0a119867c.mp4

The models recieve the baby's sleep data in the .csv format from the BabyTracker app and then process it into a binary time-series of sleep/awake status over a sequence of 10-minute intervals.

The current model ensemble consists of two LightGBM-based models and one SARIMAX model.

### LGBM 6-step averaging custom model
The first LGBM model, which is also the most accurate in the ensemble, assumes that the baby is old enough so that over the course of one hour, there's going to be only one status change, either from 'sleeping' to 'awake' or vice-versa. I then use an LGBM regressor to predict the amount of sleep over the next 60 minutes. Predictions for overlapping intervals are then combined to produce a forecast for each individual step in a time series.
Here's this model's ROC curve for 1-step prediction for a 4 to 6 month old baby:

![LGBM_avg](https://user-images.githubusercontent.com/17629106/160226908-cf09b78a-b79a-47d6-98ae-d4a59264debc.png)

This looks pretty good. However, it is important to note that upon a random selection of time intervals after which we make a 1-step prediction, these time intervals are dominated by the examples where the next step's status is equal to the previous step's status. To make sure that the model didn't simply learn to predict the previous step's status, let's take a look at the ROC curve calculated for the examples, where the next step status is different from the previous step's:

![LGBM_avg_th](https://user-images.githubusercontent.com/17629106/160227021-3976351d-32df-433f-bccd-2b3c992d48f6.png)

It is clear, that when it comes to toughest examples, our model resorts to random guessing. Thankfully, the is smart enough to recognize that it needs to stop predicting previous steps's state when it comes to these examples.

### Predicting deviations from the general trend
The other two models make use of the [NYU data set](https://cims.nyu.edu/~sl5924/infant.html) from which we extract an averaged sleep pattern for infants between 0 and 12 months old. The models then try to predict the *deviaton* from the average sleep pattern for a given infant. From the decomposition of the time series, is clear that the deviation has a strong and stable seasonal component:

![decomp](https://user-images.githubusercontent.com/17629106/160227997-b87da84a-d0a8-4abc-84c0-c34280ff280e.png)

### LGBM model predicting 1-step deivation
The other LGBM model simply predicts the next-step deviation from the average trend with the following ROC curve:

![LGBM_next_step](https://user-images.githubusercontent.com/17629106/160227232-6abf86cf-c71b-46c7-8bda-aa2db03a5f2a.png)

### SARIMAX model
The last model is a SARIMAX, which is fitted with the 'season' parameter equal to 144 (number of 10-minute intervals in a day). To estimate the *p* and *q* parameters of SARIMAX model, we looked at the ACF and PACF plots and then did a grid search to find the best parameters around the values suggested by ACF and PACF plots. A more detailed exploration of this model can be found in a /notebooks/SARIMAX_model.ipynb notebook file. 
