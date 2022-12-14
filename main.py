import numpy as np
import pandas as pd
import math
import altair as alt 
from prophet import Prophet
from prophet.diagnostics import cross_validation 
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly, plot_components_plotly

from PIL import Image
import streamlit as st 
import base64 #to open .gif files in streamlit app
from pathlib import Path 
from datetime import date, datetime 
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_absolute_error 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose 


pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500) 
pd.set_option('display.width', 800) 
st.set_option('deprecation.showPyplotGlobalUse', False) 

def img_to_bytes(img_path):
	img_bytes = Path(img_path).read_bytes() 
	encoded = base64.b64encode(img_bytes).decode() 
	return encoded 

@st.cache 
def convert_df(df):
    return df.to_csv().encode('utf-8')

# functions
@st.experimental_memo
def get_simple_model(df):
	# df.columns = ['ds','y']
	m = Prophet(interval_width=0.95)
	m.fit(df) 
	return m 
	
@st.experimental_memo(ttl=60 * 60 * 24)
def comparison_chart(data, title):
    hover = alt.selection_single(
        fields=["ds"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title=title)
        .mark_line()
        .encode(
            x="ds",
            y='y:Q',
            color="set",
        )
    )
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="ds",
            y='y:Q',
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("ds", title="ds"),
                alt.Tooltip('y:Q', title=["y"]),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


	 




# GET DATA  
uploaded_file = st.file_uploader("Upload your csv with 2 columns: 'ds' for data and 'y' for the predicting values")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=['ds'])
    data['ds'] = pd.to_datetime(data['ds'].apply(lambda x: x.date()) )
    st.write(data)
    
    # first main plot 
    st.write(' ')
    st.markdown('#### Original data')
    plt.figure(figsize=(11,4))
    df = data.reset_index()
    c = alt.Chart(df).mark_line().encode(x='ds', y='y', tooltip=['ds', 'y'])
    st.altair_chart(c, use_container_width=True)

    st.markdown('---')
    st.markdown("#### Forecast use a Simple automated Prophet model")
    df = data.reset_index(drop = True)
    min_date = df.ds.min()
    m = get_simple_model(df)
    future = m.make_future_dataframe(periods=12, freq = 'MS')
    forecast = m.predict(future) 


    # main chart
    fig = plot_plotly(m, forecast)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text='y')
    fig.update_xaxes(title_text = 'x')
    #component chart
    fig_comp = plot_components_plotly(m,forecast)
    fig_comp.update_yaxes(title_text='y')
    fig_comp.update_xaxes(title_text = 'x')

    st.write(fig, fig_comp)
	# evaluate the predictions
    st.write('#### Evaluate model predictions')
    dates = df['ds'][-4:].values
    y_true = df['y'][-4:].values
    y_pred = forecast['yhat'][-4:].values
    source = pd.DataFrame({'ds':dates, 'Actual':y_true,'Predict':y_pred})
    source.set_index('ds', inplace=True)
    source = source.reset_index().melt('ds', var_name='set', value_name='y')
	# plot
    chart = comparison_chart(source, "Simple - Prophet's prediction evaluation")
    st.altair_chart((chart).interactive(), use_container_width=True)
	# assess the model with MAE
    mae = mean_absolute_error(y_true, y_pred)
    st.success('MAE: %.3f' % mae)

    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
    st.write(forecast)

    csv = convert_df(forecast[['ds','yhat_lower','yhat','yhat_upper']])
    st.download_button(
        label = 'Download forecast as CSV'
        ,data = csv 
        ,file_name = 'forecast.csv'
        ,mime = 'text/csv'
    )
