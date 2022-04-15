"""
This streamlit app will be used to present a dashboard for data science
using the YouTube dataset available in Kaggle.
Data source: Ken Jee YouTube Data

"""
# importing common libraries
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


# feature engineering on agg dataframe
# adding additional columns
def feature_agg(dataset):
    dataset.drop(index=1, inplace=True)
    dataset['Video publish time'] = pd.to_datetime(dataset['Video publish time'])
    dataset['Average view duration'] = dataset['Average view duration'].apply(
        lambda x: datetime.strptime(x, '%H:%M:%S'))
    dataset['Average duration seconds'] = dataset['Average view duration'].apply(
        lambda x: x.second)
    dataset['Engagement ratio'] = (dataset['Comments added'] + dataset['Shares'] + dataset['Dislikes'] +
                                   dataset['Likes']) / dataset.Views
    dataset['Views / sub gained'] = dataset['Views'] / dataset['Subscribers gained']
    return dataset


# grouping the countries
def audience_simple(country):
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'


# dataframe mapping functions
def style_negative(v, props=''):
    try:
        return props if v < 0 else None
    except:
        pass


def style_positive(v, props=''):
    try:
        return props if v > 0 else None
    except:
        pass


# loading the datasets from local disk
@st.cache
def load_data():
    columns = ['Video', 'Video title', 'Video publish time', 'Comments added', 'Shares', 'Dislikes', 'Likes',
               'Subscribers lost', 'Subscribers gained', 'RPM (USD)', 'CPM (USD)', 'Average percentage viewed',
               'Average view duration', 'Views', 'Watch time (hours)', 'Subscribers', 'Your estimated revenue',
               'Impressions', 'Impressions click-through rate']
    df_agg = pd.read_csv("C:/Users/igiligi/Documents/Datasets/YouTube Data/Aggregated_Metrics_By_Video.csv")
    df_agg.columns = columns
    df_agg.drop([0], axis=0, inplace=True)
    feature_agg(df_agg)
    df_agg_sub = pd.read_csv(
        "C:/Users/igiligi/Documents/Datasets/YouTube Data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv")
    df_agg_sub.dropna(axis=0, how='any', inplace=True)
    df_comments = pd.read_csv("C:/Users/igiligi/Documents/Datasets/YouTube Data/All_Comments_Final.csv")
    df_time = pd.read_csv("C:/Users/igiligi/Documents/Datasets/YouTube Data/Video_Performance_Over_Time.csv")
    return df_agg, df_agg_sub, df_comments, df_time


agg, agg_sub, comments, time = load_data()

# develop new features of the percentage delta in views over the past 12 months
agg_diff = agg.copy()
metric_date_12mo = agg_diff['Video publish time'].max() - pd.DateOffset(months=12)
median_agg = agg_diff[agg_diff['Video publish time'] >= metric_date_12mo].median()

numeric_cols = np.array((agg_diff.dtypes == 'float64') | (agg_diff.dtypes == 'int64'))
agg_diff.iloc[:, numeric_cols] = (agg_diff.iloc[:, numeric_cols] - median_agg).div(median_agg)

# merge daily data with publish data to get delta
time_diff = pd.merge(time, agg.loc[:, ['Video', 'Video publish time']], left_on='External Video ID', right_on='Video')
time_diff['days published'] = (time_diff['Date'] - time_diff['Video publish time']).dt.day

# get last 12 months of data rather than all data
date_12mo = agg['Video publish time'].max() - pd.DateOffset(months=12)
time_diff_yr = time_diff[time_diff['Video publish time'] >= date_12mo]

# get daily view data (first 30), median & percentiles
views_days = pd.pivot_table(
    time_diff_yr,
    index='days_published',
    values='Views',
    aggfunc=[np.mean, np.median, lambda x: np.percentile(x, 80), lambda x: np.percentile(x, 20)])
views_days.columns = ['days published', 'mean views', 'median_views', '80pct views', '20pct views']
views_days = views_days[views_days['days published'].between(0, 30)]
views_cum = views_days.loc[:, ['days published', 'median views', '80pct views', '20pct views']]
views_cum.loc[:, ['median views', '80pct views', '20pct vies']] = \
    views_cum.loc[:, ['median views', '80pct views', '20pct views']].cumsum()

"""
BUILDING THE STREAMLIT DASHBOARD
"""
st.title('YouTube Dashboard: Video Engagement Analysis')
st.subheader('Analysis of views and engagement for the Ken Jee Channel')
with st.sidebar:
    select_box = st.selectbox("Aggregate or Individual Video",
                              ('Aggregate Metrics', 'Individual Video Analysis'))

# adding components to each sidebar option
if select_box == 'Aggregate Metrics':
    agg_metrics = agg[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares',
                       'Comments added', 'RPM (USD)', 'Average percentage viewed',
                       'Average duration seconds', 'Engagement ratio', 'Views / sub gained']]
    metric_date_6mo = agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo_ = agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    metric_medians6mo = agg_metrics[agg_metrics['Video publish time'] >= metric_date_6mo].median()
    metric_medians12mo = agg_metrics[agg_metrics['Video publish time'] >= metric_date_12mo_].median()

    col1, col2, col3, col4, col5 = st.columns(5)
    column = [col1, col2, col3, col4, col5]

    # adding the metric values in columns 1 - 5
    count = 0
    for i in metric_medians6mo.index:
        with column[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i]) / metric_medians12mo[i]
            st.metric(label=i, value=round(metric_medians6mo[i], 1), delta="{:.2%}".format(delta))
            count += 1
            if count >= 5:
                count = 0

    # adding and formatting the dataframe by color
    agg_diff['Publish date'] = agg_diff['Video publish time'].apply(lambda x: x.date())
    agg_diff_final = agg_diff.loc[:, ['Video title', 'Publish date', 'Views', 'Likes', 'Subscribers',
                                      'Shares', 'Comments added', 'RPM (USD)', 'Average percentage viewed',
                                      'Average duration seconds', 'Engagement ratio', 'Views / sub gained']]

    agg_numeric_list = agg_diff_final.median().index.tolist()
    df_to_pct = {}
    for i in agg_numeric_list:
        df_to_pct[i] = '{:.1%}'.format

    st.dataframe(
        agg_diff_final.style.applymap(style_negative, props='color:red').applymap(
            style_positive, props='color:green').format(df_to_pct))

elif select_box == 'Individual Video Analysis':
    videos = tuple(agg['Video title'])
    video_selected = st.selectbox('Pick a video', videos)

    filtered = agg[agg['Video title'] == video_selected]
    sub_filtered = agg_sub[agg_sub['Video Title'] == video_selected]
    sub_filtered['Country'] = [audience_simple(country=x) for x in sub_filtered['Country Code']]
    sub_filtered.sort_values('Is Subscribed', inplace=True)

    # visualization with plotly express
    fig = px.bar(data_frame=sub_filtered, x='Views', y='Is Subscribed', color='Country', orientation='h',
                 title='Bar plot of ' + video_selected)
    st.plotly_chart(fig)

    agg_time_filtered = time_diff[time_diff['Video Title'] == video_selected]
    first_30 = agg_time_filtered[agg_time_filtered['days published'].between(0, 30)]
    first_30 = first_30.sort_values('days published')

    # second visualization using graph objects
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=views_cum['days published'],
        y=views_cum['20pct views'],
        mode='lines+markers',
        name='20th percentile',
        line=dict(color='purple', dash='dash')))
    fig2.add_trace(go.Scatter(
        x=views_cum['days published'],
        y=views_cum['median views'],
        mode='lines',
        name='50th percentile',
        line=dict(color='black', dash='dash')))
    fig2.add_trace(go.Scatter(
        x=views_cum['days published'],
        y=views_cum['80pct views'],
        mode='lines',
        name='80th percentile',
        line=dict(color='royalblue', dash='dash')))
    fig2.add_trace(go.Scatter(
        x=first_30['days published'],
        y=first_30['Views'].cumsum(),
        mode='lines',
        name='Current video',
        line=dict(color='firebrick', width=8)))
    fig2.update_layout(title='View comparison first 30 days',
                       xaxis_title='Days Since Published',
                       yaxis_title='Cumulative views')

    st.plotly_chart(fig2)
