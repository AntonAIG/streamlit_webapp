'''
This is a base application for streamlit
In this app, we want to create a basic data analysis streamlit web app
A dataset from the New York motor vehicles collisions and crashes recorded from
2014 to 2020 shows the number of accidents in NYC
'''

import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from dateutil import parser
import plotly.express as px

DATA_URL = (
    "C:/Users/igiligi/Documents/Datasets/Motor_Vehicle_Collisions_-_Crashes.csv"
)
st.title("Motor Vehicle Collisions in New York City")
st.markdown("## This application is a Streamlit dashboard that can be used to "
            "analyze motor vehicles collisions in NYC 🗽 🎇 🚗")


# used to convert the column headers from upper case to lower case
def lowercase():
    return lambda x: str(x).lower()


'''def isotime(datestring):
    datestring = str(datestring)
    date = datestring.replace("T", " ")
    dto = parser.parse(date)
    return dto
'''

@st.cache(persist=True)
# building a function that loads the dataset with datetime parse
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['CRASH_DATE', 'CRASH_TIME']])
    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    data.rename(lowercase(), axis='columns', inplace=True)
    data.rename(columns={'crash_date_crash_time': 'date/time'}, inplace=True)
    return data


df = load_data(100000)
main_data = df

# new header for new questions from the data
st.header("Where are the most people injured in NYC?")
injured_people = st.slider("Number of persons injured in vehicle collisions in NYC", 0, 19)
st.map(df.query("injured_persons >= @injured_people")[["latitude", "longitude"]].dropna(how="any"))

# new header
st.header("How many collisions occur during a given time of day")
hour = st.slider("Hour to at", 0, 24)
data = df[df['date/time'].dt.hour == hour]

# vehicle collision intervals
st.markdown("Vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))
midpoint = (np.average(df['latitude']), np.average(df['longitude']))

# load the pydeck map
# we can add layers to the map
st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[pdk.Layer(
        "HexagonLayer",
        data=df[['date/time', 'latitude', 'longitude']],
        get_position=['longitude', 'latitude'],
        radius=100,
        extruded=True,
        pickable=True,
        elevation_scale=4,
        elevation_range=[0, 1000],
    ),
    ],
))

st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
filtered = df[
    (df['date/time'].dt.hour >= hour) & (df['date/time'].dt.hour > (hour + 1))
    ]  # just a dataframe selection based on two mutually inclusive conditions

# creating a histogram of the data
hist = np.histogram(filtered['date/time'].dt.minute, bins=60, range=(0, 60))[0]
chart_data = pd.DataFrame({'minute': range(60), 'crashes': hist})
fig = px.bar(chart_data, x='minute', y='crashes',
             hover_data=['minute', 'crashes'], height=400)
st.write(fig)

# select data using dropdowns
st.header("Top 5 dangerous streets by affected type")
select = st.selectbox('Affected type of people', ['Pedestrians', 'Cyclists', 'Motorists'])

if select == 'Pedestrians':
    st.write(
        main_data.query("injured_pedestrians >= 1")[["on_street_name", "injured_pedestrians"]] \
            .sort_values(by=['injured_pedestrians'], ascending=False).dropna(how='any')[:5])

elif select == 'Cyclists':
    st.write(
        main_data.query("injured_cyclists >= 1")[["on_street_name", "injured_cyclists"]] \
            .sort_values(by=['injured_cyclists'], ascending=False).dropna(how='any')[:5])

else:
    st.write(
        main_data.query("injured_motorists >= 1")[["on_street_name", "injured_motorists"]] \
            .sort_values(by=['injured_motorists'], ascending=False).dropna(how='any')[:5])

# toggle bottom to show raw data
if st.checkbox("Show Raw Data", False):
    st.subheader('Raw Data')
    st.write(df)
