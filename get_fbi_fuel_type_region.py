# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:34:39 2024

@author: clark
"""
import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
from datetime import datetime, timedelta
import time
import sys, os
from fbi_vic_plot_functions import plot_fbi_and_rating_with_fwas

def get_fuel_type_fbi_in_a_region(fbi_arr, areas_shapefile, area_name, fuel_map, fuel_lut, fuel_model):
        #Clip fbi_arr to the area given by area_name
        area_polygon = areas_shapefile[areas_shapefile['Area_Name']==area_name]
        fbi_arr.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
        fbi_arr.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
        fbi_area_clipped = fbi_arr.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)

        #Filter fuel map down to this area as well.
        fuel_clipped = xr.where(fbi_area_clipped, fuel_map, np.nan)
        
        #Now filter to the fuel type given by fuel_model.
        fuel_codes_trimmed = fuel_lut[fuel_lut['FBM']==fuel_model]['FTno_State'].values
        fbi_area_fuel_clipped = fbi_area_clipped.where(fuel_clipped.isin(fuel_codes_trimmed))
        
        return fbi_area_fuel_clipped

if __name__=="__main__":
    #Set dates:
        year_sel_ = 2024
        mon_sel = 2
        day_sel = 21

        datetime_sel = datetime(year_sel_, mon_sel, day_sel)

        forecast_day = 1
        datetime_fc = datetime_sel + timedelta(days=forecast_day)
    
        #set strings here - bit of a mess but helps later!
        #Note - we add nothing if we want day+1, bc UTC puts us to next day
        mon_sel_str = datetime.strftime(datetime_sel, "%m")
        day_sel_str = datetime.strftime(datetime_sel, "%d")
        mon_sel_str_fc = datetime.strftime((datetime_sel+timedelta(days=forecast_day-1)), "%m")
        day_sel_str_fc = datetime.strftime((datetime_sel+timedelta(days=forecast_day-1)), "%d")
        day_sel_str_fcplus1 = datetime.strftime((datetime_sel+timedelta(days=forecast_day)), "%d")
        mon_sel_str_fcplus1 = datetime.strftime((datetime_sel+timedelta(days=forecast_day)), "%m")


        #load the file:
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_feb_mar23/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_control/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
        
        #get maximum:
        start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
        end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
        #start_ind=3
        start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
        end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
        daily_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:]
        max_recalc_fbi = daily_fbi.max(dim='time', keep_attrs=True)
        max_recalc_rating = recalc_file_in['rating_1'][start_ind:end_ind,:,:].max(dim='time',keep_attrs=True)
        
        #Get shapefile and fuel lookup table:
#        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp")
        area_name = 'Pyrenees'
        
        #Get fuel lookup table:
        fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv"
        fuel_lut_in = pd.read_csv(fuel_lut_path)
        fuel_model = 'Pine'
        
        #Get fuel map:
        fuel_map_rc = recalc_file_in['fuel_type'][12,:,:]  #first index is arbitrary, perhaps don't use 0 just in case...
        
        #Now reduce the data to only that fuel model in that region:
        fbi_clipped_to_fuel = get_fuel_type_fbi_in_a_region(max_recalc_fbi, shp_in, area_name, fuel_map_rc, fuel_lut_in, fuel_model)
        rating_clipped_to_fuel = xr.where(~np.isnan(fbi_clipped_to_fuel), max_recalc_rating, np.nan)
        
        plot_fbi_and_rating_with_fwas(fbi_clipped_to_fuel, rating_clipped_to_fuel, shp_in, box_extent=[142.9,143.9,-38.,-36.6])
        
        fbi_table_ = fbi_clipped_to_fuel.to_dataframe().dropna(subset='index_1')
        
        #Let's get all the variables for the max index from the recalc.
        daily_fbi = daily_fbi.fillna(-99)  #just fill it with a "silly" negative value to stop it from being unhappy
        max_time = daily_fbi.argmax(dim='time', keep_attrs=True)
        max_temp = recalc_file_in['T_SFC'][start_ind:end_ind,:,:].isel({'time': max_time})
        min_rh = recalc_file_in['RH_SFC'][start_ind:end_ind,:,:].isel({'time': max_time})
        df_at_max = recalc_file_in['DF_SFC'][start_ind:end_ind,:,:].isel({'time': max_time})
        kbdi_at_max = recalc_file_in['KBDI_SFC'][start_ind:end_ind,:,:].isel({'time': max_time})
        wind_at_max = recalc_file_in['WindMagKmh_SFC'][start_ind:end_ind,:,:].isel({'time': max_time})
        
        temp_clipped = xr.where(~np.isnan(fbi_clipped_to_fuel), max_temp, np.nan)
        temp_clipped.name = 'T_SFC'
        rh_clipped = xr.where(~np.isnan(fbi_clipped_to_fuel), min_rh, np.nan)
        rh_clipped.name = 'RH_SFC'
        df_clipped = xr.where(~np.isnan(fbi_clipped_to_fuel), df_at_max, np.nan)
        df_clipped.name = 'DF_SFC'
        kbdi_clipped = xr.where(~np.isnan(fbi_clipped_to_fuel), kbdi_at_max, np.nan)
        kbdi_clipped.name = 'KBDI_SFC'
        wind_clipped = xr.where(~np.isnan(fbi_clipped_to_fuel), wind_at_max, np.nan)
        wind_clipped.name = 'WindMagKmh_SFC'
        
        temp_table = temp_clipped.to_dataframe().dropna(subset='T_SFC').drop(columns=['band','spatial_ref'])
        final_table = pd.merge(fbi_table_, temp_table, how='inner', left_index=True, right_index=True)
        rh_table = rh_clipped.to_dataframe().dropna(subset='RH_SFC').drop(columns=['band','spatial_ref'])
        kbdi_table = kbdi_clipped.to_dataframe().dropna(subset='KBDI_SFC').drop(columns=['band','spatial_ref'])
        df_table = df_clipped.to_dataframe().dropna(subset='DF_SFC').drop(columns=['band','spatial_ref'])
        wind_table = wind_clipped.to_dataframe().dropna(subset='WindMagKmh_SFC').drop(columns=['band','spatial_ref'])
        final_table = pd.merge(final_table, rh_table, how='inner', left_index=True, right_index=True)
        final_table = pd.merge(final_table, wind_table, how='inner', left_index=True, right_index=True)
        final_table = pd.merge(final_table, df_table, how='inner', left_index=True, right_index=True)
        final_table = pd.merge(final_table, kbdi_table, how='inner', left_index=True, right_index=True)
        
        