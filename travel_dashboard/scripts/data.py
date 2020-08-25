from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests


# Script to read in data and prepare the plotly visualizations.

def get_coviddata(country):
    """Computes new covid cases over last 14 days for a country

    Args:
        string: ISO 3166-1 alpha-2 format of country code

    Returns:
        dataframe: dataframe containing new cases timeseries

    """
    # pull data from api
    url = 'http://corona-api.com/countries/{}'.format(country)
    try:
        r = requests.get(url)
        data = r.json()
    except:
        print('could not load data')

    # country population
    population = data['data']['population']

    # filter and sort values
    timeline = data['data']['timeline']
    df = pd.DataFrame(timeline)
    df = df[['date', 'new_confirmed']]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'], inplace=True)
    df.reset_index(inplace=True)

    # normalization and rolling sum of new cases
    df['new_100k'] = df['new_confirmed'] * 10 ** 5 / population
    df['new_100k_sum14d'] = df['new_100k'].rolling(14).sum()

    return df


def get_fx():
    """Return EUR/CHF daily timeseries from 1st jan 2020

    Args:
        None

    Returns:
        dataframe: dataframe containing EUR/CHF timeseries

    """
    # pull data from api
    url = 'https://api.exchangeratesapi.io/history?start_at=2020-01-01&end_at={}&symbols=CHF'.format(
        datetime.today().strftime('%Y-%m-%d'))
    try:
        r = requests.get(url)
        data = r.json()
    except:
        print('could not load data')

    df = pd.DataFrame.from_dict(data['rates'], orient='index')

    return df


def return_figures():
    """Creates four plotly visualizations

    Args:
        None

    Returns:
        list (dict): list containing the four plotly visualizations

    """

    # first chart plots rolling sum of new cases for GR and CH
    # as a line chart

    df_gr = get_coviddata('GR')
    df_ch = get_coviddata('CH')

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_ch.date, y=df_ch.new_100k_sum14d, name='Switzerland'))
    fig1.add_trace(go.Scatter(x=df_gr.date, y=df_gr.new_100k_sum14d, name='Greece'))
    fig1.add_trace(go.Scatter(x=df_ch.date, y=[60] * len(df_ch.date), name='Threshold',
                              line=dict(color='firebrick', dash='dash')
                              ))
    fig1.update_layout(title='Number of new infections in the past 14 days',
                       title_x=0.5,
                       xaxis_title='Month',
                       yaxis_title='Per 100 000 persons',
                       legend=dict(
                           yanchor="top",
                           y=0.99,
                           xanchor="right",
                           x=0.99,
                           bgcolor='rgba(0,0,0,0)'
                       )
                       )

    # second chart plots EUR/CHF rate
    # as a line chart
    df_fx = get_fx()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_fx.index, y=df_fx.CHF))
    fig2.update_layout(title='Exchange Rate',
                       title_x=0.5,
                       xaxis_title='Month',
                       yaxis_title='EUR/CHF')

    # append all charts to the figures list
    figures = [fig1, fig2]

    return figures
