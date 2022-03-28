import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Text
from os import getenv
from sys import exit
import zipfile
import src.data_processing.load_data as load_data
import src.data_processing.prediction_processing as predprocess
import src.models.LGBM_sleepy_score as sscore
import src.models.ensemble as ensemble
import src.models.SARIMAX as SARIMAXfc
import numpy as np

bot_token = getenv('BOT_TOKEN')
if not bot_token:
    exit('Error: no token provided')

bot = Bot(token=bot_token)

# Bot dispatcher + loging
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)


@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await message.reply('Hello! I can tell you how sleepy your baby is based '
                        'on the sleeping data you collected via BabyTracker '
                        'app. I can also learn your baby\'s sleep patterns '
                        'and suggest a natural sleeping schedule over the '
                        'next 24 hours for them.')
    await bot.send_message(chat_id=message.from_user.id,
                           text= 'Please forward me CSV data from '
                                 'the app to get my estimate.')


@dp.message_handler(content_types=[types.ContentType.DOCUMENT])
async def download_doc(message: types.Message):
    # download user data
    await message.document.download(destination_file='../data/raw/csv.zip')
    with zipfile.ZipFile('../data/raw/csv.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/raw/csv/')

    #process the data
    sleep, time0 = load_data.get_sleep('12/06/2021', '../data/')

    # generate sleepy score prediction
    rawscore = sscore.sleepy_score(sleep)
    if rawscore < 2.5:
        score = 0
    elif rawscore < 5:
        score = 1
    else:
        score = 2
    sleepdict = ['Not sleepy at all!', 'Somewhat sleepy.', 'Very sleepy!']

    await bot.send_message(chat_id=message.from_user.id,
        text='Sleepy score: {p:1d} out of 3'.format(p = score + 1),
        parse_mode='Markdown')

    await bot.send_message(chat_id=message.from_user.id,
                           text=sleepdict[score])

    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["Yes", "No"]
    keyboard.add(*buttons)
    await message.answer('Would you like me to generate a suggested '
                         'sleeping schedule for the next 24 hours?',
                         reply_markup=keyboard)

@dp.message_handler(Text(equals='Yes'))
async def yes_generate(message: types.Message):
    await message.reply('Give me about 5 to 10 minutes...',
                        reply_markup=types.ReplyKeyboardRemove())

    # process the data
    sleep, time0 = load_data.get_sleep('12/06/2021', '../data/')

    # define smoothing function which helps to average over several predictions
    weights = np.array([1, 1, 1])
    conv = np.array([1, 1, 2, 4, 6, 4, 2, 1, 1])
    conv = conv / conv.sum()

    # define needed variables
    forecast_period = 1 * 24 * 6 + 30
    DATA_PATH = '../data/'
    MODEL_PATH = '../src/models/'

    # compute the sleep prediction
    ans, pred1, pred2, pred3 = ensemble.forecast(sleep, time0, DATA_PATH,
                                                 MODEL_PATH, forecast_period,
                                                 weights, conv, conv)

    # generate plots of the main forecast, as well
    # as the plots for individual models' predictions
    predprocess.plot_predict(ans, '../predictions/',
                             'forecast.png')
    predprocess.plot_predict(pred1, '../predictions/',
                             'prediction_LGBM_1step.png')
    predprocess.plot_predict(pred2, '../predictions/',
                             'prediction_LGBM_avg.png')
    predprocess.plot_predict(pred3, '../predictions/',
                             'prediction_SARIMAX.png')

    await bot.send_photo(chat_id = message.from_user.id,
                         photo=open('../predictions/forecast.png', 'rb'))

    # update the model if needed
    SARIMAXfc.model_update(sleep, time0, MODEL_PATH)


@dp.message_handler(Text(equals='No'))
async def dont_generate(message: types.Message):
    await message.reply("All right! See you soon!",
                        reply_markup=types.ReplyKeyboardRemove())
    # process the data
    sleep, time0 = load_data.get_sleep('12/06/2021', '../data/')
    MODEL_PATH = '../src/models/'

    # update the model if needed
    SARIMAXfc.model_update(sleep, time0, MODEL_PATH)

@dp.message_handler(commands='update')
async def cmd_start(message: types.Message):
    await message.reply('Begin updating the SARIMAX '
                        'model with the most recent data')

    # update the model
    sleep, time0 = load_data.get_sleep('12/06/2021', '../data/')
    MODEL_PATH = '../src/models/'
    SARIMAXfc.model_update(sleep, time0, MODEL_PATH, force_update=True)

    await bot.send_message(chat_id=message.from_user.id,
                           text='The model updated successfully!')


if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)