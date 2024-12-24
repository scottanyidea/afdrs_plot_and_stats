"""
Calculate FBI from AWS data that has been compiled from an archive.
This allows one to specify the fuel type model used.
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
import fdrs_calcs 

import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":
    #Read AWS compiled table:
#    table_in = pd.read_csv("C://Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/Active Projects/AFDRS Research - Eval/EVALUATION TASKS/FDI_FBIforPDD/PDD22_23/vic_aws_mar23_apr23.csv",
#                           dtype={'Station_full': 'str', 'Station_desc': 'str', 'Primary FBM': 'str', 'Secondary FBM': 'str'},
#                           parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
    table_in = pd.read_csv("C://Users/clark/analysis1/compiled_obs/compiled_obs_westernvic_20241117.csv",
                           dtype={'station_full': 'str', 'station_desc': 'str', 'primary FBM': 'str', 'secondary FBM': 'str'},
                           parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
#    table_in = table_in[table_in['station_full']=='MALLACOOTA']
#    table_in['time'] = table_in['time']-timedelta(hours=11)
    
    #Set default grass condition to grazed - edit later if needed...
    default_grass_cond = 2
    
    #Get fuel lookup table and set fuel types we want to calculate:
#    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv")
    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-generic.csv")
    fuel_type_table = pd.read_excel("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/obs_station_fuel_types_VIC.xlsx")
    fuel_type_ = [4000]

    #Output table same as in... plus some columns to be calculated.
    table_out = table_in
    
    #Quick modification to DF - this is optional and should be commented out mostly:
    #table_in['DF']= 9.5

    """
    #Loop over fuel types to calculate FBI:
    #If fuel_type_ above is empty, this whole loop is skipped.
    print("Calculating FBIs")
    for ft in fuel_type_:
        fuel_line = fuel_lut[fuel_lut['FTno_State']==ft]
        
        calculated_fdrs_output_np_arr = fdrs_calcs.calculate_indicies(
                temp = table_in['temperature'].values,
                kbdi = table_in['KBDI'].values,
                sdi = np.full(len(table_in), np.nan),
                windmag10m = table_in['wind speed kmh'].values,
                rh = table_in['RH'].values,
                td = table_in['dew point'].values,
                df = table_in['DF'].values,
                curing = table_in['curing'].values,
                grass_fuel_load = table_in['grass fuel load'].values,
                precip = table_in['accum precip'].values,
                time_since_rain = np.full(len(table_in), np.nan),
                time_since_fire = np.full(len(table_in), 25),
                ground_moisture = np.full(len(table_in), np.nan),
                fuel_type = np.full(len(table_in), ft),
                fuel_table = fuel_line,
                #hours = np.full(len(table_in), 0),
                hours = table_in['time'].dt.hour.values,
                months = table_in['time'].dt.month.values,
                grass_condition = np.full(len(table_in), default_grass_cond))

        #Get FBI, rate of spread and intensity for now. Maybe we want others later??
        table_out['FBI_'+str(ft)] = calculated_fdrs_output_np_arr['index_1']
        
    """
    
    #OK we want FFDI and GFDI too. Let's calculate those.
    
    # GFDI
    print("Calculate GFDI using the BoM Specification version")
    GFDI_SFC=np.round(        
            np.exp(
                -1.523
                + 1.027 * np.log(table_in['grass fuel load'].values)
                - 0.009432 * np.power((100 - table_in['curing'].values), 1.536)
                + 0.02764 * table_in['temperature'].values 
                + 0.6422 * np.power(table_in['wind speed kmh'].values, 0.5) 
                - 0.2205 * np.power(table_in['RH'].values, 0.5)
            )
        )
        
    #FFDI:
    print("Calculate FFDI")
    
    FFDI_SFC= np.round(
                2 * np.exp(-0.45 + 0.987 * np.log(table_in['DF'].values) - 0.0345 * table_in['RH'].values + 0.0338 * table_in['temperature'].values + 0.0234 * table_in['wind speed kmh'])
            )
    
    table_out['FFDI'] = FFDI_SFC
    table_out['GFDI'] = GFDI_SFC
    #Save:
    table_out = table_out.rename(columns={'Time': 'time', 'Station_full': 'station_full', 'Latitude': 'latitude', 'Longitude': 'longitude',
                                  'Temperature': 'temperature', 'Dew point': 'dew point', 'Wind dir': 'wind dir', 'Wind speed': 'wind speed', 'Wind gust': 'wind gust',
                                  'Curing': 'curing', 'Grass Fuel Load': 'grass fuel load'})
    table_out = table_out.drop(columns=['Unnamed: 0','station_desc'])
    #table_out.to_csv("C://Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/Active Projects/AFDRS Research - Eval/EVALUATION TASKS/FDI_FBI comparison for PDD/PDD22_23/vic_aws_mar23_apr23_fdis.csv", index=False)
    table_out.to_csv("C:/Users/clark/analysis1/compiled_obs/obs_20241116_otways_ffdi.csv")
    #Calculate also the maximums throughout the day.
    #TODO: Fix this by sorting by FBI then grouping by station. At the moment
    #this takes just the maximum of each column. Or... is this really what we want???
    table_out_max = table_out.groupby('station_full', as_index=False).max('Primary FBI')
    table_out_max.to_csv('C:/Users/clark/analysis1/compiled_obs/obs_20241116_otways_ffdi_maximums.csv', index=False)