#Plot daily maxima of AWS observations 
#Designed for a multi day dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

#Load data:
data_in = pd.read_csv("C:/Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data/obs_huntershill_20260111_20260213_fbicalcs.csv", parse_dates=['time'], date_format='mixed', dayfirst=True)

#Calculate wind dir in deg:
bearing_dict = {'N': 0.,
                'NNE': 22.5,
                'NE': 45.,
                'ENE': 67.5,
                'E': 90.,
                'ESE': 112.5,
                'SE': 135.,
                'SSE': 157.5,
                'S': 180.,
                'SSW': 202.5,
                'SW': 225.,
                'WSW': 247.5,
                'W': 270.,
                'WNW': 292.5,
                'NW': 315.,
                'NNW': 337.5}

data_in['wind dir deg'] =data_in['wind dir']
data_in['wind dir deg'] = data_in['wind dir deg'].map(bearing_dict)

#Get maxima by date:
data_in['date'] = pd.to_datetime(data_in['time'].dt.date)
max_ffdi = data_in.loc[data_in.groupby('date')['FFDI'].idxmax()]

#formatting parameters - set up windows for lower and upper y lims of plots
temp_upper = np.nanmax([30, max(max_ffdi['temperature']+2)])
temp_lower = np.nanmin([10, min(max_ffdi['temperature']-2)])
fdi_lower = 0
fdi_upper = np.nanmax([50, max(max_ffdi['FFDI']+5)])

time_lower = max_ffdi.date.min()
time_upper = max_ffdi.date.max()


#Plot:
fig, axs= plt.subplots(3,1,figsize=(6,12))

axs[0].plot(max_ffdi.date, max_ffdi.temperature, color='k')
axs[1].plot(max_ffdi.date, max_ffdi['wind speed kmh'], color='k', label='speed')
axs[1].plot(max_ffdi.date, max_ffdi['wind gust kmh'], color='cornflowerblue', label='gust')
x0 = md.date2num(max_ffdi.date)  #convert to numbers which refers to days since... well, something...
y0 = np.zeros(len(x0)) + 5
dx = -np.sin(np.pi/180.*max_ffdi['wind dir deg'].values) * 1.25
dy = -np.cos(np.pi/180. * max_ffdi['wind dir deg'].values) * 4
x0_arr = x0 - 0.5*dx
y0 = y0 - 0.5* dy
for i in range(0, len(x0)):
    arr = plt.Arrow(x0_arr[i], y0[i], dx[i], dy[i], width=0.5, edgecolor='black')
    axs[1].add_patch(arr)
axs[2].plot(max_ffdi.date, max_ffdi.FFDI, color='k')

axs[0].set_xticklabels([])
axs[0].set_ylim(temp_lower, temp_upper)
axs[0].set_ylabel('Temperature (deg C)', fontsize=12)
axs[0].yaxis.set_tick_params(labelsize=12)
axs[0].set_xlim(time_lower, time_upper)
axs[0].set_xticks(x0)
axs[0].grid()
axs[1].legend()
axs[1].set_xticklabels([])
axs[1].set_ylabel('Wind (km/h)')
axs[1].yaxis.set_tick_params(labelsize=12)
axs[1].set_xlim(time_lower, time_upper)
axs[1].set_xticks(x0)
axs[1].grid()
axs[2].set_ylim(fdi_lower, fdi_upper)
axs[2].set_ylabel('FFDI', fontsize=12)
axs[2].yaxis.set_tick_params(labelsize=12)
axs[2].set_xlim(time_lower, time_upper)
axs[2].set_xticks(x0)
axs[2].xaxis.set_tick_params(rotation=90, labelsize=12)
axs[2].grid()

fig.suptitle('Hunters Hill AWS (time of max FFDI)', fontsize=16)
fig.tight_layout()

fig.savefig('./Case_studies/2026_01_09/compiled_weather_data/huntershill_maxffdi_20260111_20260213')
