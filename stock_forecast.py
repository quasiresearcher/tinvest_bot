import tinvest
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from sklearn.linear_model import LinearRegression
from tinvest import TooManyRequestsError
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from suppress_prophet_out import *
from tqdm import tqdm

token = '''t.njIVOO-Z4q5kQNfoBTMsAmLb0UPZxbYw42xEbvG9ZeStCvx38Xw2dLahXw02AX_uD77lBqCcjbfOhDuhMgCadQ'''

client = tinvest.SyncClient(token, use_sandbox=True)

# давайте сразу сюда добавим вывод всех возможных инструментов
instruments = pd.DataFrame(client.get_market_stocks().payload.instruments)
instruments = df_column_renamer(instruments)
instruments['currency'] = instruments['currency'].str.split('.').str[-1]
instruments['type'] = instruments['type'].str.split('.').str[-1]
instruments.drop(['isin', 'lot', 'min_price_increment', 'min_quantity'], axis=1, inplace=True)

output = pd.DataFrame()

for i in tqdm(range(len(instruments))):

    try:
        response = client.get_market_candles(from_=(pd.to_datetime('today') - pd.Timedelta('365 days')),
                                             to=pd.to_datetime('today'),
                                             figi=instruments.iloc[i]['figi'],
                                             interval=tinvest.CandleResolution.day)
    except TooManyRequestsError:
        print('Хватит с тебя запросов')
        break

    stock_df = pd.DataFrame(response.payload.candles)
    if len(stock_df) < 1:
        continue

    name = instruments.iloc[i]['name']
    currency = instruments.iloc[i]['currency']
    one_stock = pd.DataFrame({'Name': name,
                              'Currency': currency,
                              'Graph name': name + '.jpg'}, index=[len(output)])
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.title(name)
    plt.ylabel(currency)
    plt.xlabel('Time')
    columns_to_predict = 'high'
    stock_df = df_column_renamer(stock_df)

    stock_df.rename(columns={'c': 'close',
                             'l': 'low',
                             'h': 'high',
                             'o': 'open'},
                    inplace=True)

    stock_df[['close', 'high', 'low', 'open']] = stock_df[['close', 'high', 'low',
                                                           'open']].astype(float)

    stock_df['days'] = [i for i in range(1, len(stock_df) + 1)]
    # если тренд идёт горизонтально или уходит вниз, то можно завязывать
    reg = LinearRegression().fit(stock_df[['days']], stock_df[columns_to_predict])
    plt.plot(stock_df['time'], reg.predict(stock_df[['days']]), color="blue")
    stock_df.set_index('time')[[columns_to_predict]].plot(ax=ax, color='blue', alpha=0.8)

    if reg.coef_[0] < 0:

        plt.savefig('graphs/' + name + '.jpg')
        one_stock['Conclusion'] = 'Тренд плохой'
        output = output.append(one_stock)
        continue

    stock_df.drop(['interval', 'v'], axis=1, inplace=True)
    stock_df['time'] = stock_df['time'].dt.tz_localize(None)

    predictions = 30

    stock_df.rename(columns={columns_to_predict: 'y', 'time': 'ds'}, inplace=True)
    stock_df = stock_df[['ds', 'y']]

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    cutoffs = pd.to_datetime([stock_df['ds'].iloc[int(len(stock_df) * 0.4)],
                              stock_df['ds'].iloc[int(len(stock_df) * 0.7)]])

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        with suppress_stdout_stderr():
            m = Prophet(**params).fit(stock_df)
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days')
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params = all_params[np.argmin(rmses)]

    with suppress_stdout_stderr():
        m = Prophet(**best_params).fit(stock_df)
        future = m.make_future_dataframe(periods=predictions)
        forecast = m.predict(future)

    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast.rename(columns={'yhat': 'forecast',
                             'yhat_lower': 'pessimistic forecast',
                             'yhat_upper': 'optimistic forecast'},
                    inplace=True)
    stock_df.rename(columns={'y': 'actual data'},
                    inplace=True)

    forecast_from = stock_df['ds'].max()

    stock_df = pd.merge(stock_df, forecast, on='ds', how='left').drop('forecast', axis=1)
    forecast = forecast[forecast['ds'] >= forecast_from][['ds', 'forecast', 'pessimistic forecast',
                                                          'optimistic forecast']]
    forecast.loc[forecast['ds'] == forecast_from, 'forecast'] = stock_df.query('ds == @forecast_from'
                                                                                ).iloc[-1]['actual data']
    stock_df.set_index('ds', inplace=True)
    forecast.set_index('ds', inplace=True)

    forecast['days'] = [i for i in range(1, len(forecast) + 1)]
    reg = LinearRegression().fit(forecast[['days']], forecast['forecast'])
    plt.plot(forecast.index, reg.predict(forecast[['days']]), color="red", alpha=0.4)
    stock_df[['pessimistic forecast', 'optimistic forecast']].plot(ax=ax, color='blue', alpha=0.2)
    ax.fill_between(x=stock_df.index, y1=stock_df['pessimistic forecast'], y2=stock_df['optimistic forecast'],
                    alpha=0.05, color='blue')

    ax.fill_between(x=forecast.index, y1=forecast['pessimistic forecast'], y2=forecast['optimistic forecast'],
                    alpha=0.05, color='red')

    forecast[['forecast']].plot(ax=ax, color='red', alpha=0.8)
    forecast[['pessimistic forecast', 'optimistic forecast']].plot(ax=ax, color='red', alpha=0.2)

    ax.legend(handles=[Line2D([0], [0], color='blue', label='Actual'),
                       Line2D([0], [0], color='red', label='Prophet')])

    plt.savefig('graphs/' + name + '.jpg')
    if 0 < reg.coef_[0]:

        one_stock['Conclusion'] = 'Прогноз норм'
        output = output.append(one_stock)

    else:

        one_stock['Conclusion'] = 'Прогноз плохм'
        output = output.append(one_stock)

    if i > 10:
        break

output.to_excel('total_conclusion.xlsx', index=False)
