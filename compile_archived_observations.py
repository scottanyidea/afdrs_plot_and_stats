"""
I finally found where the AWS observations live on the archive (embarrassingly!)
So let's function this to compile together a table of observations.

TODO: Where no stations are wanted, filter the archive down to Victorian stations.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import gzip
import json
import sys, os
from pathlib import Path

def make_observation_table_from_archive(start_timestamp, end_timestamp, stations=None):
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
        current_time_str_folder = (current_time+timedelta(hours=11)).strftime(format("%Y_%m_%d"))
        print('Starting '+current_time_str)
        #See if we can get the exact time step for every 10 minutes.
        #More often than not, there is instead a file with a time step ending in a 9,
        #ie. 1 minute before the 10 minute block... so just sneakily grab that.
        #If not, just assume that time step is missing.
        try:
            #Need to read the file as bytes, decompress them, then decode them using utf-8,
            #then we have a huge series of bytes. Then json loads formats it nicely.
            with open('M://Archived/BoM_AWS_OBS_IDZ20081_ARCHIVE/'+current_time_str_folder+'/IDZ20081_current_obs.json.'+current_time_str+'Z.gz', 'rb') as read_gz_:
                file_data_in = json.loads(gzip.decompress(read_gz_.read()).decode('utf-8'))
            current_time_list.append(current_time+timedelta(hours=11))  #add it back in local time...
        except FileNotFoundError:
            print("***Could not find this, try the stamp one minute earlier...")
            time_diff_ = 0
            while time_diff_ <= 5:
                try:
                    current_time_temp = current_time-timedelta(minutes=1)
                    time_diff_ = time_diff_ + 1
                    current_time_str = current_time_temp.strftime(format("%Y%m%dT%H%M"))
                    with open('M://Archived/BoM_AWS_OBS_IDZ20081_ARCHIVE/'+current_time_str_folder+'/IDZ20081_current_obs.json.'+current_time_str+'Z.gz', 'rb') as read_gz_:
                        file_data_in = json.loads(gzip.decompress(read_gz_.read()).decode('utf-8'))
                    current_time_list.append(current_time_temp+timedelta(hours=11))
                    print("***Success!")
                    break
                except FileNotFoundError:
                    print("***Still not finding, one more step...")
                    continue
                except ValueError:
                    print('There was an error decoding the JSON. Skipping.')
                    break
            if time_diff_==5:
                print("OK I think it's not here. Just go to the next time stamp.")
            current_time = current_time+timedelta(minutes=10)
            continue
    
        national_data = pd.json_normalize(file_data_in['data'])
        #Check there's actually data here. Some of the files are empty...
        if national_data.empty:
            #Reset current time 10 minutes later
            print('This file is empty. Go to the next.')
            current_time = current_time+timedelta(minutes=10)
            k=k+1
            continue

        #Now grab our desired rows.
        if stations is not None:
            row_data = national_data.loc[national_data['station_info.station_name'].isin(stations)][[
                            'station_info.bom_id',
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
                            'observation_data.grass_fuel_load',
                            'observation_data.df',
                            'observation_data.upper_level_soil_moisture_fullness',
                            'station_info.primary_fbm',
                            'observation_data.primary_fbi',
                            'station_info.secondary_fbm',
                            'observation_data.secondary_fbi']]
        else:
            row_data = national_data[[
                            'station_info.bom_id',
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
                            'observation_data.grass_fuel_load',
                            'observation_data.df',
                            'observation_data.upper_level_soil_moisture_fullness',
                            'station_info.primary_fbm',
                            'observation_data.primary_fbi',
                            'station_info.secondary_fbm',
                            'observation_data.secondary_fbi']]
        row_data['Time'] = current_time_list[-1]
        #If first iteration, start the pandas dataframe. If not, just append it.
        if k==0:
            output_data = row_data
        else:
            output_data = output_data._append(row_data, ignore_index=True)
    
        #Reset current time 10 minutes later
        current_time = current_time+timedelta(minutes=10)
        k=k+1

    #Don't forget to add the timestamps... and place it as the first row.
    output_data = output_data[['Time'] + [col for col in output_data.columns if col!='Time']]
    #Rename columns to stuff we want:
    output_data.columns=   ['time',
                            'bom_id',
                            'station_full',
                            'station_desc',
                            'latitude',
                            'longitude',
                            'temperature',
                            'RH',
                            'dew point',
                            'wind dir',
                            'wind speed kmh',
                            'wind gust kmh',
                            'accum precip',
                            'KBDI',
                            'curing',
                            'grass fuel load',
                            'DF',
                            'upper_soil_fullness',
                            'primary FBM',
                            'primary FBI',
                            'secondary FBM',
                            'secondary FBI']

    
    
    return output_data

"""
Main function
"""
if __name__=='__main__':
    start_date = datetime(year=2024,month=9,day=1,hour=6,minute=0,second=0)
    end_date = datetime(year=2025, month=5,day=1,hour=23,minute=55,second=59)
    """
    stations_to_pick = ["NHILL AERODROME", "HORSHAM AERODROME", "EDENHOPE AIRPORT"]
    """
    """
    stations_to_pick = ["AIREYS INLET", "MOUNT GELLIBRAND", "MILDURA AIRPORT", "WALPEUP RESEARCH", "HORSHAM AERODROME",
                        "NHILL AERODROME", "KANAGULK", "CASTERTON"]
    """
    """
    stations_to_pick = ['MILDURA AIRPORT', 'WALPEUP RESEARCH', 'HOPETOUN AIRPORT',
                        'NHILL AERODROME', 'HORSHAM AERODROME', 'EDENHOPE AIRPORT',
                        'DARTMOOR', 'WESTMERE', 'MORTLAKE RACECOURSE',
                        'KYABRAM','SHEPPARTON AIRPORT', 'BENDIGO AIRPORT',
                        'WANGARATTA AERO', 'ALBURY AIRPORT AWS', 'FALLS CREEK',
                        'KILMORE GAP', 'EILDON FIRE TOWER',
                        'BALLARAT AERODROME', 'MOORABBIN AIRPORT', 'GEELONG RACECOURSE',
                        'NILMA NORTH (WARRAGUL)', 'LATROBE VALLEY AIRPORT', 'EAST SALE AIRPORT',
                        'MOUNT NOWA NOWA', 'ORBBOST', 'OMEO']
    
    """    
    stations_to_pick = ["AIREYS INLET", "ALBURY AIRPORT AWS", "AVALON AIRPORT", "BAIRNSDALE AIRPORT",
                        "BALLARAT AERODROME", "BEN NEVIS", "BENDIGO AIRPORT", "CAPE NELSON LIGHTHOUSE", 
                        "CAPE OTWAY LIGHTHOUSE", "CASTERTON", "CERBERUS", "CHARLTON", 
                        "COLDSTREAM", "COMBIENBAR AWS", "DARTMOOR", "EAST SALE AIRPORT", 
                        "EILDON FIRE TOWER", "EDENHOPE AIRPORT", "ESSENDON AIRPORT", 
                        "FALLS CREEK", "FERNY CREEK", "GELANTIPY", 
                        "GEELONG RACECOURSE", "HUNTERS HILL", "HAMILTON AIRPORT","HOPETOUN AIRPORT", 
                        "HORSHAM AERODROME","KILMORE GAP", "KANAGULK", 
                        "KYABRAM", "LATROBE VALLEY AIRPORT", "LAVERTON RAAF", "LONGERENONG", 
                        "MALLACOOTA","MANGALORE AIRPORT", "MELBOURNE AIRPORT", "MILDURA AIRPORT", 
                        "MOORABBIN AIRPORT",
                        "MOUNT BULLER", "MOUNT BAW BAW", "MOUNT GELLIBRAND", "MOUNT HOTHAM AIRPORT", 
                        "MOUNT WILLIAM", 
                        "MOUNT NOWA NOWA", "MORTLAKE RACECOURSE", "NILMA NORTH (WARRAGUL)", 
                        "NHILL AERODROME", "RUTHERGLEN RESEARCH",
                        "MOUNT MOORNAPA", "OMEO", "ORBOST", "PORT FAIRY AWS", "PORTLAND NTC AWS", 
                        "REDESDALE",
                        "SCORESBY RESEARCH INSTITUTE", "SHEOAKS", "SHEPPARTON AIRPORT", "STAWELL AERODROME",
                        "SWAN HILL AERODROME", "TATURA INST SUSTAINABLE AG", "VIEWBANK", "WALPEUP RESEARCH", 
                        "WANGARATTA AERO", "WARRACKNABEAL AIRPORT", "WARRNAMBOOL AIRPORT NDB", "WESTMERE", 
                        "WILSONS PROMONTORY LIGHTHOUSE", "YARRAWONGA"]
    
    output_table = make_observation_table_from_archive(start_date, end_date, stations_to_pick)
    print(str(len(stations_to_pick))+' stations chosen in this calc')
    print(str(len(output_table['bom_id'].unique()))+' stations found.')
#    output_table.to_csv('C:/Users/clark/analysis1/compiled_obs/compiled_obs_statesample_20250203-20250204.csv')
    output_table.to_csv("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/Active Projects/AFDRS Research - Eval/EVALUATION TASKS/FDI_FBIforPDD/PDD24_25/vic_aws_sep24_apr25.csv")
