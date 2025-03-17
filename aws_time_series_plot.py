#Plot time series of AWS data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the data
#aws_data_in = pd.read_csv('C:/Users/clark/analysis1/compiled_obs/compiled_obs_statesample_20250203-20250204.csv', parse_dates=['time'])
aws_data_in = pd.read_csv('C:/Users/clark/analysis1/compiled_obs/obs_statesample_20250203-20250204_fdis.csv', parse_dates=['time'])

#choose the data to plot:
#stations_to_choose = ['MILDURA AIRPORT', 'WALPEUP RESEARCH', 'HOPETOUN AIRPORT']
#stations_to_choose = ['NHILL AERODROME', 'HORSHAM AERODROME', 'EDENHOPE AIRPORT']
#stations_to_choose = ['DARTMOOR', 'WESTMERE', 'MORTLAKE RACECOURSE']
#stations_to_choose = ['KYABRAM','SHEPPARTON AIRPORT', 'BENDIGO AIRPORT']
#stations_to_choose = ['WANGARATTA AERO', 'ALBURY AIRPORT AWS', 'FALLS CREEK']
#stations_to_choose=['KILMORE GAP', 'EILDON FIRE TOWER']
#stations_to_choose=['BALLARAT AERODROME', 'MOORABBIN AIRPORT', 'GEELONG RACECOURSE']
stations_to_choose = ['NILMA NORTH (WARRAGUL)', 'LATROBE VALLEY AIRPORT', 'EAST SALE AIRPORT']

#plot:
fig, axs = plt.subplots(1)
fig2, axs2 = plt.subplots(1)
fig3, axs3 = plt.subplots(1)
for i in range(0, len(stations_to_choose)):
    aws_data_subset = aws_data_in[aws_data_in['station_full']==stations_to_choose[i]]
    axs.plot(aws_data_subset['time'], aws_data_subset['primary FBI'], label=stations_to_choose[i])
    axs2.plot(aws_data_subset['time'], aws_data_subset['wind speed kmh'], label=stations_to_choose[i])
    axs3.plot(aws_data_subset['time'], aws_data_subset['FFDI'], label=stations_to_choose[i])
axs.tick_params(axis='x', labelrotation=90)
axs.set_ylabel('FBI')
axs.set_xlabel('Time MM-DD HH')
axs.legend()
axs2.tick_params(axis='x', labelrotation=90)
axs2.set_ylabel('Wind speed')
axs2.legend()
axs2.set_xlabel('Time MM-DD HH')
axs3.tick_params(axis='x', labelrotation=90)
axs3.set_ylabel('FFDI')
axs3.legend()
axs3.set_xlabel('Time MM-DD HH')

