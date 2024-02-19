"""
I finally found where the AWS observations live on the archive (embarrassingly!)
So let's function this to compile together a table of observations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gzip
import json
import sys, os
from pathlib import Path

def make_observation_table_from_archive(start_timestamp, end_timestamp, stations):
    #Set up current time for the loop.
    #First we need to go to UTC...
    start_date_z = start_timestamp-timedelta(hours=11)
    end_date_z = end_timestamp-timedelta(hours=11)

    current_time = start_date_z
    k=0
    current_time_list = []  #save the current time in the file...

    #Iterate through every time step in the archive from start_date to end_date.
    while current_time <= end_date_z:
        current_time_str = current_time.strftime(format("%Y%m%dT%H%M"))
        print('Starting '+current_time_str)
        #See if we can get the exact time step for every 10 minutes.
        #More often than not, there is instead a file with a time step ending in a 9,
        #ie. 1 minute before the 10 minute block... so just sneakily grab that.
        #If not, just assume that time step is missing.
        try:
            #Need to read the file as bytes, decompress them, then decode them using utf-8,
            #then we have a huge series of bytes. Then json loads formats it nicely.
            with open('M://Archived/BoM_AWS_OBS_IDZ20081_ARCHIVE/IDZ20081_current_obs.json.'+current_time_str+'Z.gz', 'rb') as read_gz_:
                file_data_in = json.loads(gzip.decompress(read_gz_.read()).decode('utf-8'))
            current_time_list.append(current_time+timedelta(hours=11))  #add it back in local time...
        except FileNotFoundError:
            print("***Could not find this, try the stamp one minute earlier...")
            try:
                current_time_temp = current_time-timedelta(minutes=1)
                current_time_str = current_time_temp.strftime(format("%Y%m%dT%H%M"))
                with open('M://Archived/BoM_AWS_OBS_IDZ20081_ARCHIVE/IDZ20081_current_obs.json.'+current_time_str+'Z.gz', 'rb') as read_gz_:
                    file_data_in = json.loads(gzip.decompress(read_gz_.read()).decode('utf-8'))
                current_time_list.append(current_time_temp+timedelta(hours=11))
            except FileNotFoundError:
                print("OK I think it's not here. Just go to the next time stamp.")
                current_time = current_time+timedelta(minutes=10)
                continue
    
        national_data = pd.json_normalize(file_data_in['data'])

        #Now grab our desired rows.
        row_data = national_data.loc[national_data['station_info.station_name'].isin(stations_to_pick)][[
                            'station_info.station_name',
                            'station_info.description',
                            'station_info.latitude',
                            'station_info.longitude',
                            'observation_data.temp',
                            'observation_data.rh',
                            'observation_data.dewt',
                            'observation_data.wind_dir',
                            'observation_data.wnd_spd_kmh',
                            'observation_data.wnd_gust_spd_kmh',
                            'observation_data.accumulated_precip',
                            'observation_data.kbdi',
                            'observation_data.curing',
                            'observation_data.df',
                            'station_info.primary_fbm',
                            'observation_data.primary_fbi',
                            'station_info.secondary_fbm',
                            'observation_data.secondary_fbi']]
    
        #If first iteration, start the pandas dataframe. If not, just append it.
        if k==0:
            output_data = row_data
        else:
            output_data.loc[k] = row_data.iloc[0]
    
        #Reset current time 10 minutes later
        current_time = current_time+timedelta(minutes=10)
        k=k+1


    output_data.columns=   ['Station_full',
                                   'Station_desc',
                                   'Latitude',
                                   'Longitude',
                                   'Temperature',
                                   'RH',
                                   'Dew point',
                                   'Wind dir',
                                   'Wind speed kmh',
                                   'Wind gust kmh',
                                   'Accum precip',
                                   'KBDI',
                                   'Curing',
                                   'DF',
                                   'Primary FBM',
                                   'Primary FBI',
                                   'Secondary FBM',
                                   'Secondary FBI']

    #Don't forget to add the timestamps... and place it as the first row.
    output_data['Time'] = current_time_list
    output_data = output_data[['Time'] + [col for col in output_data.columns if col!='Time']]
    return output_data

"""
Main function
"""
if __name__=='__main__':
    start_date = datetime(year=2024,month=2,day=13,hour=0,minute=0,second=0)
    end_date = datetime(year=2024, month=2,day=13, hour=23,minute=55,second=59)
    
    stations_to_pick = ['BALLARAT AERODROME']

    output_table = make_observation_table_from_archive(start_date, end_date, stations_to_pick)

output_table.to_csv('Archived_obs_20240213_Ballarat.csv')