"""
Calculate FBI from AWS data that has been compiled from an archive.
This allows one to specify the fuel type model used.

To use: - only need to change the obs table used in table_in, and ensure the fuel
lut is up to date. Can use multiple fuel sub-types for comparison, just add to the
list at fuel_type_.
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
import fdrs_calcs 

import warnings
warnings.filterwarnings("ignore")


def calculate_fbi_for_fueltype(table_input, fuel_lut, fuel_types, grass_condition=2, time_since_fire=25, time_since_rain=48):
    k=0
    for ft in fuel_types:
        fuel_line = fuel_lut[fuel_lut['FTno_State']==ft]
        
        calculated_fdrs_output_np_arr = fdrs_calcs.calculate_indicies(
                temp = table_input['temperature'].values,
                kbdi = table_input['KBDI'].values,
                sdi = np.full(len(table_input), np.nan),
                windmag10m = table_input['wind speed kmh'].values,
                rh = table_input['RH'].values,
                td = table_input['dew point'].values,
                df = table_input['DF'].values,
                curing = table_input['curing'].values,
                grass_fuel_load = table_input['grass fuel load'].values,
                precip = np.full(len(table_input), 0),
                time_since_rain = np.full(len(table_input), 48),
                time_since_fire = np.full(len(table_input), 25),
                ground_moisture = np.full(len(table_input), np.nan),
                fuel_type = np.full(len(table_input), ft),
                fuel_table = fuel_line,
                #hours = np.full(len(table_in), 0),
                hours = table_input['time'].dt.hour.values,
                months = table_input['time'].dt.month.values,
                grass_condition = np.full(len(table_input), grass_condition))

        if k==0:
            cols_out = pd.DataFrame(calculated_fdrs_output_np_arr['index_1'], columns=['FBI_'+str(ft)])
            k=k+1  #for first pass, create dataframe to contain output columns. Else, just add them to the DF
        else:
            cols_out['FBI_'+str(ft)] = calculated_fdrs_output_np_arr['index_1']
        cols_out['ROS_'+str(ft)] = calculated_fdrs_output_np_arr['rate_of_spread']

    return cols_out

def calculate_gfdi_for_awsdata(table_input):
    GFDI_SFC=np.round(        
            np.exp(
                -1.523
                + 1.027 * np.log(table_input['grass fuel load'].values)
                - 0.009432 * np.power((100 - table_input['curing'].values), 1.536)
                + 0.02764 * table_input['temperature'].values 
                + 0.6422 * np.power(table_input['wind speed kmh'].values, 0.5) 
                - 0.2205 * np.power(table_input['RH'].values, 0.5)
            )
        )
    return GFDI_SFC

def calculate_ffdi_for_awsdata(table_input):
    """
    Calculate FFDI from AWS data in pandas DF input.

    Parameters
    ----------
    table_input : Pandas dataframe containing the data. Must contain columns
    labelled "DF" (drought factor), 'RH' (humidity), 'temperature', and 'wind 
    speed kmh'. Any other columns in the table will not be used.

    Returns
    -------
    FFDI_SFC : Pandas series containing calculated FFDI.

    """
    FFDI_SFC= np.round(
                2 * np.exp(-0.45 + 0.987 * np.log(table_input['DF'].values) - 0.0345 * table_input['RH'].values + 0.0338 * table_input['temperature'].values + 0.0234 * table_input['wind speed kmh'])
            )
    return FFDI_SFC

if __name__=="__main__":
    #Read AWS compiled table:
#    table_in = pd.read_csv("C://Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/Active Projects/AFDRS Research - Eval/EVALUATION TASKS/FDI_FBIforPDD/PDD22_23/vic_aws_mar23_apr23.csv",
#                           dtype={'Station_full': 'str', 'Station_desc': 'str', 'Primary FBM': 'str', 'Secondary FBM': 'str'},
#                           parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
    table_in = pd.read_csv("C://Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data/obs_albury_20260111_20260213.csv",
                           dtype={'station_full': 'str', 'station_desc': 'str', 'primary FBM': 'str', 'secondary FBM': 'str'},
                           parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
    table_in.time = pd.to_datetime(table_in.time)
#    table_in = table_in[table_in['station_full']=='MALLACOOTA']
#    table_in['time'] = table_in['time']-timedelta(hours=11)
    
    #Set default grass condition to grazed - edit later if needed...
    default_grass_cond = 2
    
    #Get fuel lookup table and set fuel types we want to calculate:
    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv")
#    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-generic.csv")
#    fuel_type_ = [3020, 3007]
    fuel_type_ = []

    #Output table same as in... plus some columns to be calculated.
    table_out = table_in
    
    #Quick modification to DF - this is optional and should be commented out mostly:
    #table_in['DF']= 9.5

    #Modification for curing - alternative values:
    #table_in['curing'] = 90

    #Loop over fuel types to calculate FBI:
    #If fuel_type_ above is empty, this whole loop is skipped.
    print("Calculating FBIs")

#    cols_fbi_calc = calculate_fbi_for_fueltype(table_in, fuel_lut, fuel_type_)    
#    table_out = pd.concat([table_out, cols_fbi_calc], axis=1)
    
    #OK we want FFDI and GFDI too. Let's calculate those.
    
    # GFDI
    print("Calculate GFDI using the BoM Specification version")
    GFDI_SFC = calculate_gfdi_for_awsdata(table_in)

    #FFDI:
    print("Calculate FFDI")

    FFDI_SFC = calculate_ffdi_for_awsdata(table_in)
    table_out['FFDI'] = FFDI_SFC
    table_out['GFDI'] = GFDI_SFC
    
    #Save:
    table_out = table_out.rename(columns={'Time': 'time', 'Station_full': 'station_full', 'Latitude': 'latitude', 'Longitude': 'longitude',
                                  'Temperature': 'temperature', 'Dew point': 'dew point', 'Wind dir': 'wind dir', 'Wind speed': 'wind speed', 'Wind gust': 'wind gust',
                                  'Curing': 'curing', 'Grass Fuel Load': 'grass fuel load'})
    table_out = table_out.drop(columns=['Unnamed: 0','station_desc'])
    #table_out.to_csv("C://Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/Active Projects/AFDRS Research - Eval/EVALUATION TASKS/FDI_FBI comparison for PDD/PDD22_23/vic_aws_mar23_apr23_fdis.csv", index=False)
    table_out.to_csv("C:/Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data/obs_albury_20260111_20260213_fbicalcs.csv")
    """
    #Calculate also the maximums throughout the day.
    #TODO: Fix this by sorting by FBI then grouping by station. At the moment
    #this takes just the maximum of each column. Or... is this really what we want???
    table_out_max = table_out.groupby('station_full', as_index=False).max('Primary FBI')
    table_out_max.to_csv('C:/Users/clark/analysis1/compiled_obs/obs_statesample_20250203-20250204_fdismax.csv', index=False)
    """