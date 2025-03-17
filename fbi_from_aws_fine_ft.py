"""
Calculate FBI from AWS data that has been compiled from an archive.
This allows one to specify the fuel type model used.
"""
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
#    table_in = pd.read_csv("C://Users/clark/analysis1/compiled_obs/compiled_obs_202324.csv",
#                           dtype={'station_full': 'str', 'station_desc': 'str', 'primary FBM': 'str', 'secondary FBM': 'str'},
#                           parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
    table_in = pd.read_csv("C://Users/clark/analysis1/compiled_obs/AWS_fine_fuels/obs_202324_recalc_forest.csv",
                           dtype={'station_full': 'str', 'station_desc': 'str', 'primary FBM': 'str', 'secondary FBM': 'str'},
                           parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')


    #Set default grass condition to grazed - edit later if needed...
    default_grass_cond = 2
    
    #Get fuel lookup table and set fuel types we want to calculate:
#    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv")
    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-generic.csv")
    fuel_type_table = pd.read_excel("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/obs_station_fuel_types_VIC.xlsx")
#    fuel_type_table = fuel_type_table[['bom-id', 'station_name', 'primary_fine_fuel_type_code', 'secondary_fine_fuel_type_code']]
    fuel_type_table = fuel_type_table[['bom-id', 'station_name', 'NEW PRIMARY FUEL TYPE CODE', 'NEW SECONDARY FUEL TYPE CODE']]
    table_matched_fine_fuels = pd.merge(table_in, fuel_type_table, left_on='bom_id', right_on='bom-id')
    table_matched_fine_fuels = table_matched_fine_fuels.rename(columns={'NEW PRIMARY FUEL TYPE CODE': 'primary_fine_fuel_type_code', 'NEW SECONDARY FUEL TYPE CODE': 'secondary_fine_fuel_type_code'})

    
    fuel_type_ = np.concatenate([table_matched_fine_fuels['primary_fine_fuel_type_code'].unique(),table_matched_fine_fuels['secondary_fine_fuel_type_code'].unique()])

    #Output table same as in... plus some columns to be calculated.
    table_out = table_matched_fine_fuels
    
    #Quick modification to DF - this is optional and should be commented out mostly:
    #table_in['DF']= 9.5

    #set up arrays for calcing index, ROS, etc
    fbi_out_primary = np.full(len(table_matched_fine_fuels), np.nan)
    fbi_out_secondary = np.full(len(table_matched_fine_fuels), np.nan)
    
    #Loop over fuel types to calculate FBI:
    #If fuel_type_ above is empty, this whole loop is skipped.
    print("Calculating FBIs")
    for ft in fuel_type_:
        #Loop through each fuel type in the fuel type list.
        fuel_line = fuel_lut[fuel_lut['FTno_State']==ft]
        
        calculated_fdrs_output_np_arr = fdrs_calcs.calculate_indicies(
                temp = table_out['temperature'].values,
                kbdi = table_out['KBDI'].values,
                sdi = np.full(len(table_out), np.nan),
                windmag10m = table_out['wind speed kmh'].values,
                rh = table_out['RH'].values,
                td = table_out['dew point'].values,
                df = table_out['DF'].values,
                curing = table_out['curing'].values,
                grass_fuel_load = table_out['grass fuel load'].values,
                precip = table_out['accum precip'].values,
                time_since_rain = np.full(len(table_out), 3),
                time_since_fire = np.full(len(table_out), 25),
                ground_moisture = np.full(len(table_out), np.nan),
                fuel_type = np.full(len(table_out), ft),
                fuel_table = fuel_line,
                hours = table_out['time'].dt.hour.values,
                months = table_out['time'].dt.month.values,
                grass_condition = np.full(len(table_out), default_grass_cond))

        #Assign the values only to those rows for which FT is the primary.
        primary_fbi_mask = table_matched_fine_fuels['primary_fine_fuel_type_code']==ft
        fbi_out_primary[primary_fbi_mask] = calculated_fdrs_output_np_arr['index_1'][primary_fbi_mask]
        
        #...and same for secondary.
        sec_fbi_mask = table_matched_fine_fuels['secondary_fine_fuel_type_code']==ft
        fbi_out_secondary[sec_fbi_mask] = calculated_fdrs_output_np_arr['index_1'][sec_fbi_mask]

    #OK we want FFDI and GFDI too. Let's calculate those.
    
    table_out['primary_fine_FBI'] = fbi_out_primary
    table_out['secondary_fine_FBI'] = fbi_out_secondary
    
    
    # GFDI
    print("Calculate GFDI using the BoM Specification version")
    GFDI_SFC=np.round(        
            np.exp(
                -1.523
                + 1.027 * np.log(table_out['grass fuel load'].values)
                - 0.009432 * np.power((100 - table_out['curing'].values), 1.536)
                + 0.02764 * table_out['temperature'].values 
                + 0.6422 * np.power(table_out['wind speed kmh'].values, 0.5) 
                - 0.2205 * np.power(table_out['RH'].values, 0.5)
            )
        )
        
    #FFDI:
    print("Calculate FFDI")
    
    FFDI_SFC= np.round(
                2 * np.exp(-0.45 + 0.987 * np.log(table_out['DF'].values) - 0.0345 * table_out['RH'].values + 0.0338 * table_out['temperature'].values + 0.0234 * table_out['wind speed kmh'])
            )
    
    table_out['FFDI'] = FFDI_SFC
    table_out['GFDI'] = GFDI_SFC
    
    table_out = table_out.rename(columns={'Time': 'time', 'Station_full': 'station_full', 'Latitude': 'latitude', 'Longitude': 'longitude',
                                  'Temperature': 'temperature', 'Dew point': 'dew point', 'Wind dir': 'wind dir', 'Wind speed': 'wind speed', 'Wind gust': 'wind gust',
                                  'Curing': 'curing', 'Grass Fuel Load': 'grass fuel load'})
#    table_out.to_csv("C:/Users/clark/analysis1/compiled_obs/obs_202324_fine_fbis.csv", index=False)
    
    #Calculate also the maximums throughout the day, for all fuel types.
    #First let's tidy the table a bit.
    table_out = table_out.drop(columns=['Unnamed: 0', 'bom-id', 'station_name'])

    #Save whole table so we have all the weather data.
#    print('Saving full data')
#    table_out.to_csv("C:/Users/clark/analysis1/compiled_obs/obs_202324_fine_fbis_all_3020_3007.csv", index=False)
    
    print('Calculating daily max by station')
    table_out['date'] = table_out['time'].dt.date
    #For FBMs - all time steps have the same FBM for a given station. So just grab the first.
    table_out_max = table_out.groupby(['station_full', 'date'], as_index=False).agg({'primary FBM': 'first', 'primary FBI': 'max', 
                                                                                     'secondary FBM': 'first', 'secondary FBI': 'max', 
                                                                                     'primary_fine_fuel_type_code': 'first', 'primary_fine_FBI': 'max', 
                                                                                     'secondary_fine_fuel_type_code': 'first', 
                                                                                     'secondary_fine_FBI': 'max', 
                                                                                     'FBI_4000': 'max',
                                                                                     'FFDI': 'max', 'GFDI': 'max'})
    table_out_max = table_out_max.rename(columns={'FBI_4000': 'Generic_forest_FBI_recalc'})
    print('Saving')
    table_out_max.to_csv('C:/Users/clark/analysis1/compiled_obs/AWS_fine_fuels/obs_202324_fine_fbis_max_genericcalc_3007_MH.csv', index=False)