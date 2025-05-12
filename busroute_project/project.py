# project.py

# Neil Dewan
import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'
from plotly.colors import qualitative 

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")

""" Helper Functions"""

def convert_minutes_to_time(arrival_time):
    hour = int(arrival_time//60)
    minute = int(arrival_time % 60)
    second = int((arrival_time - (hour * 60) - minute) * 60)

    time = f"{hour:02d}:{minute:02d}:{second:02d}"
    return time

    #converts hhmmss to minutes
def time_to_minutes(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 60 + m + s / 60

# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def create_detailed_schedule(schedule, stops, trips, bus_lines):

    schedule_stops = pd.merge(schedule, stops, on='stop_id', how='inner')

    detailed_schedule = pd.merge(schedule_stops, trips, on='trip_id', how='inner')

    detailed_schedule['route_id'] = pd.Categorical(
        detailed_schedule['route_id'], categories=bus_lines, ordered=True
    )

    filtered_schedule = detailed_schedule[detailed_schedule['route_id'].notna()]

    filtered_schedule['trip_length'] = filtered_schedule.groupby('trip_id').size()

    sorted_schedule = filtered_schedule.sort_values(
        by=['route_id', 'trip_length', 'stop_sequence'], ascending=[True, True, True]
    ).drop('trip_length', axis=1).set_index('trip_id')

    return sorted_schedule




def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    unique_routes = bus_df['route_id'].unique()
    color_palette = px.colors.qualitative.Plotly # Use Plotly's qualitative color palette
    route_colors = {route: color_palette[i % len(color_palette)] for i, route in enumerate(unique_routes)}

    for route_id in unique_routes:
    # Filter the DataFrame for the current route
        filtered_bus_df = bus_df[bus_df['route_id'] == route_id]

    # Add Scattermapbox trace for the current route
        fig.add_trace(go.Scattermapbox(
            lat=filtered_bus_df['stop_lat'],
            lon=filtered_bus_df['stop_lon'],
            mode='markers',  
            marker=dict(size=8, color=route_colors[route_id]),  
            text=filtered_bus_df.apply(
            lambda row: f"({row['stop_lat']}, {row['stop_lon']})<br>{row['stop_name']}", axis=1
        ),  
        hoverinfo='text',  # Changed this line
        name=f"Bus Line {route_id}" 
    ))
    return fig


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
    filtered_df = detailed_schedule[detailed_schedule['stop_name'] == station_name]

    next_stations = np.array([], dtype=str)

    for route_id, route_data in filtered_df.groupby('route_id'):
        for trip_id in route_data.index.unique():
            trip_schedule = detailed_schedule.loc[trip_id]

            # Handle case where trip_schedule is a Series
            if isinstance(trip_schedule, pd.Series):
                trip_schedule = trip_schedule.to_frame().T

            trip_schedule = trip_schedule.sort_values('stop_sequence')

            current_stop = trip_schedule[trip_schedule['stop_name'] == station_name]['stop_sequence'].values[0]

            if current_stop < trip_schedule['stop_sequence'].max():
                next_stop = trip_schedule[trip_schedule['stop_sequence'] == current_stop + 1]['stop_name'].values[0]

                if next_stop not in next_stations:
                    next_stations = np.append(next_stations, next_stop)
    
    return next_stations


def bfs(start_station, end_station, detailed_schedule):

    # Check if stations exist in the data
    if start_station not in detailed_schedule['stop_name'].values:
        return f"Start station '{start_station}' not found."
    if end_station not in detailed_schedule['stop_name'].values:
        return f"End station '{end_station}' not found."

    # case start == end
    if start_station == end_station:
        #create a df to keep track of stops
        result_df = detailed_schedule[detailed_schedule['stop_name'] == start_station].copy()
        result_df['stop_num'] = 1
        return result_df[['stop_name', 'stop_lat', 'stop_lon', 'stop_num']]

    # Initialize BFS
    # queue entry: (current_station, path so far)
    queue = deque([(start_station, [start_station])])
    visited_stations = set()

    # 4. Loop until queue is empty or we find the end station
    while queue:
        current_station, path = queue.popleft()
        
        # check if current station is the end station
        if current_station == end_station:
            # Create a DataFrame of all stops 
            result_df = detailed_schedule[detailed_schedule['stop_name'].isin(path)].copy()
            
            # get rid of duplicates and keep the first one
            result_df = result_df.drop_duplicates(subset=['stop_name'], keep='first')
            
            # assign numbers to each station in the path in a dict
            stop_num_map = {station: i+1 for i, station in enumerate(path)}

            # map the dict to create a new col in the df with the stop num
            result_df['stop_num'] = result_df['stop_name'].map(stop_num_map)
            
            # Sort by the stop_num
            result_df = result_df.sort_values('stop_num')
            
            return result_df[['stop_name', 'stop_lat', 'stop_lon', 'stop_num']]

        # add current station to visited set
        visited_stations.add(current_station)

        # find neighbors of the current station
        neighbors = find_neighbors(current_station, detailed_schedule)

        # iterate through each neighbor and add to queue if not already visited
        for neighbor in neighbors:
            if neighbor not in visited_stations:
                queue.append((neighbor, path + [neighbor]))

    # if no path exists
    return "No path found"


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    number_of_busses = (1440-360)/tau

    arrival_times = np.sort(np.random.uniform(360, 1440, size = int(number_of_busses)))

    out = pd.DataFrame({'Arrival Time' : arrival_times})

    out['Interval'] = out['Arrival Time'].diff().fillna(out['Arrival Time'].values[0] - 360)
    out['Interval'] = np.round(out['Interval'], 2)
    out['Arrival Time'] = out['Arrival Time'].transform(convert_minutes_to_time)

    return out




# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def simulate_wait_times(arrival_times_df, n_passengers):

    #change 'Arrival Time to minutes'
    arrival_times_df["Arrival Time"] = arrival_times_df['Arrival Time'].apply(time_to_minutes)

    #set lists to keep track of values
    bus_indicies = []
    bus_arrival_times = []
    wait_times = []
    start_time = 360
    end_time = arrival_times_df['Arrival Time'].iloc[-1]

    #sort randomly generated arrival times
    sorted_arrival_times = np.sort(np.random.rand(n_passengers) * (end_time - start_time) + start_time)

    #iterate through each passengers arrival time 
    for passenger_arrival_time in sorted_arrival_times:

        #iterate through an enumerated bus arrivals to have access to each busses index
        for i, bus_arrival_time in enumerate(arrival_times_df['Arrival Time']):

            #check which bus a passenger will be allotted to
             if bus_arrival_time >= passenger_arrival_time:

                bus_indicies.append(i)

                bus_arrival_times.append(arrival_times_df['Arrival Time'].iloc[i])

                wait_times.append(bus_arrival_time - passenger_arrival_time)

                #break if a bus is found and move on to next passengers time
                break
    
    #add columns to df
    result_df = pd.DataFrame({'Passenger Arrival Time' : sorted_arrival_times, 'Bus Arrival Time': bus_arrival_times,
        'Bus Index': bus_indicies, 'Wait Time': wait_times})
    
    result_df['Passenger Arrival Time'] = result_df['Passenger Arrival Time'].apply(convert_minutes_to_time)

    result_df['Bus Arrival Time'] = result_df['Bus Arrival Time'].apply(convert_minutes_to_time)

    result_df['Bus Index'] = result_df['Bus Index'].apply(int)

    return result_df





def visualize_wait_times(wait_times_df, timestamp):
    wait_times_df['Passenger Arrival Time'] = pd.to_datetime(wait_times_df['Passenger Arrival Time'])
    wait_times_df['Bus Arrival Time']       = pd.to_datetime(wait_times_df['Bus Arrival Time'])
    
    start_time = pd.to_datetime(timestamp)
    end_time   = start_time + pd.Timedelta(hours=1)
    
    mask = (wait_times_df['Passenger Arrival Time'] >= start_time) & \
           (wait_times_df['Passenger Arrival Time'] < end_time)
    filtered_df = wait_times_df[mask].copy()
    
    bus_mask = (filtered_df['Bus Arrival Time'] >= start_time) & \
               (filtered_df['Bus Arrival Time'] <= end_time)
    
    bus_arrivals = filtered_df[bus_mask][['Bus Arrival Time','Bus Index']].drop_duplicates(
        subset='Bus Index'
    ).sort_values(by='Bus Arrival Time').copy()
    
    filtered_df['Minutes'] = (
        (filtered_df['Passenger Arrival Time'] - start_time)
        .dt.total_seconds() / 60.0
    )
    
    bus_arrivals['Minutes'] = (
        (bus_arrivals['Bus Arrival Time'] - start_time)
        .dt.total_seconds() / 60.0
    )
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=bus_arrivals['Minutes'],
            y=[0] * len(bus_arrivals),
            mode='markers',
            marker=dict(color='blue', size=10),
            name='Buses'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=filtered_df['Minutes'],
            y=filtered_df['Wait Time'],
            mode='markers',
            marker=dict(color='red', size=6),
            name='Passengers'
        )
    )
    
    for _, row in filtered_df.iterrows():
        fig.add_shape(
            type="line",
            x0=row['Minutes'], x1=row['Minutes'],
            y0=0,            y1=row['Wait Time'],
            line=dict(color='red', dash='dot', width=2)
        )
    
    fig.update_layout(
        title='Passenger Wait Times in a 60-Minute Block',
        xaxis_title='Time (minutes) within the block',
        yaxis_title='Wait Time (minutes)',
        xaxis=dict(range=[0, 60]),  
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


