import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import sklearn
import numpy as np
import plotly
from functools import reduce

#https://www.kaggle.com/datasets/kanchana1990/un-global-water-data-2012-2022
#data retrieved from Kaggle UN water dataset; modified in excel to omit unnecessary/redundant columns
water = pd.read_csv("water.csv")
water['INDICATOR'] = water['INDICATOR'].str[3:]
water

###creating independent arrays for each column and UN indicator in order to rework the dataframe to create
###independent columns for each indicator - this will help in reading the data as a table + help with using the
###data with Dash & plotly 
indicators = water['INDICATOR'].unique()
countries = water['GEOGRAPHIC_AREA'].unique()
years = water['YEAR'].unique()

ind1 = water[water['INDICATOR'] == indicators[0]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[0]}, axis='columns')
ind2 = water[water['INDICATOR'] == indicators[1]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[1]}, axis='columns')
ind3 = water[water['INDICATOR'] == indicators[2]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[2]}, axis='columns')
ind4 = water[water['INDICATOR'] == indicators[3]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[3]}, axis='columns')
ind5 = water[water['INDICATOR'] == indicators[4]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[4]}, axis='columns')
ind6 = water[water['INDICATOR'] == indicators[5]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[5]}, axis='columns')
ind7 = water[water['INDICATOR'] == indicators[6]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[6]}, axis='columns')
ind8 = water[water['INDICATOR'] == indicators[7]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[7]}, axis='columns')
ind9 = water[water['INDICATOR'] == indicators[8]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[8]}, axis='columns')
ind10 = water[water['INDICATOR'] == indicators[9]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[9]}, axis='columns')
ind11 = water[water['INDICATOR'] == indicators[10]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[10]}, axis='columns')
ind12 = water[water['INDICATOR'] == indicators[11]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[11]}, axis='columns')
ind13 = water[water['INDICATOR'] == indicators[12]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[12]}, axis='columns')
ind14 = water[water['INDICATOR'] == indicators[13]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[13]}, axis='columns')
ind15 = water[water['INDICATOR'] == indicators[14]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[14]}, axis='columns')
ind16 = water[water['INDICATOR'] == indicators[15]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[15]}, axis='columns')
ind17 = water[water['INDICATOR'] == indicators[16]].drop("INDICATOR", axis=1).rename({"PROPORTION": indicators[16]}, axis='columns')

#setting up merging columns in new df
Country = []
for i in range(0, len(countries)):
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])
    Country.append(countries[i])

Year = []
for i in range(0, len(countries)):
    Year.append(2012)
    Year.append(2013)
    Year.append(2014)
    Year.append(2015)
    Year.append(2016)
    Year.append(2017)
    Year.append(2018)
    Year.append(2019)
    Year.append(2020)
    Year.append(2021)
    Year.append(2022)

#check
len(Year) == len(Country)

water_transpose = pd.DataFrame({"GEOGRAPHIC_AREA": Country, "YEAR": Year})

ind_dfs = [water_transpose, ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8, ind9, ind10, ind11, ind12, ind13, ind14, ind15, ind16]

#merging step to finalize new df
water_transpose = reduce(lambda  left, right: pd.merge(left, right, how="left", on=["GEOGRAPHIC_AREA", "YEAR"]), ind_dfs)
water_transpose['ISO_CODE'] = water_transpose['GEOGRAPHIC_AREA'].str[:3]
water_transpose['GEOGRAPHIC_AREA'] = water_transpose['GEOGRAPHIC_AREA'].str[5:]

#pd.DataFrame.to_csv(water_transpose, 'water_transpose.csv', index=False)


#load relevant libraries for dash & plotly
import plotly.express as px
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
import json


# init data
df = water_transpose
cols_dd = df.columns[2:-1] #indicator columns

#setup app & layout, including year filter and indicator column filter dropdowns 
app = dash.Dash()
app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': k, 'value': k} for k in years],
                value=years[0],
                style={'width':'80%'}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='indicator-dropdown',
                options=[{'label': k, 'value': k} for k in cols_dd],
                value=cols_dd[0],
                style={'width':'80%'},
                optionHeight=50
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    html.Hr(),
    dcc.Graph(id='display-selected-values')
])

#define app inputs & outputs, and function to output & update map visualization
@app.callback(
    dash.dependencies.Output('display-selected-values', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')],
    [dash.dependencies.Input('indicator-dropdown', 'value')])
def update_output(year_value, indicator_value):
    dff = df[df['YEAR'] == year_value]
    
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locationmode='ISO-3',
        locations=dff['ISO_CODE'],
        z=dff[indicator_value].astype(float),
        hovertext=dff['GEOGRAPHIC_AREA'],
        colorbar_title='Proportion (%)'))
    fig.update_layout(title=f"<b>{indicator_value}</b>", title_x=0.5,
                      height=700)
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
