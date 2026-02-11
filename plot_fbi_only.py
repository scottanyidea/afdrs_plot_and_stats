#Plots the outputs to the recalculated fire danger grids, compare to official outputs

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import geopandas
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import mapping
from fbi_stat_functions import find_dominant_fuel_type_for_a_rating, find_dominant_fuel_code_for_a_rating
from fbi_vic_plot_functions import plot_fbi_with_areas
    
if __name__=="__main__":
    #Set dates:
        year_sel_ = 2026
        mon_sel = 2
        day_sel = 4

        datetime_sel = datetime(year_sel_, mon_sel, day_sel)

        forecast_day = 0
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
        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_control/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_changed_grass/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/grass_curing_20240226/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/fixed_df/df_95/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/grass_curing_20240226/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc_AM.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/IDZ10133_AUS_AFDRS_max_fbi_prelim_SFC.nc")
        """
        Find the maximum FBI and FDI at each point: 
        Note - there is a need to grab the correct time window, midnight to midnight LOCAL time.    
        """
        start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
        end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
        start_ind=3
#        start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
        end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
        max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
#        max_recalc_fbi = recalc_file_in['FDI_SFC'][:,:,start_ind:end_ind].max(dim='time',keep_attrs=True)

        """Load fire weather area (FWA) shapefile for plotting"""
        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
#        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp")
#        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90309_VIC_Boundary_SHP_ICC\PID90309_VIC_Boundary_SHP_ICC.shp")

        #extent = [140.8,145,-37.8,-33.8]    #Mallee and Wimmera
        #    extent=[140.8,143.8,-37.6,-35.3]   #Wimmera
        #    extent= [140.8,145.7,-39,-33.8]   #Wimmera + SW
        #    extent= [140.8,144.7,-38.9,-36.7]   #South West
        extent = [140.8,150,-39.3,-33.8]  #most of Vic
        #    extent= [147.0,150,-38.1,-36.4]   #East Gippsland


        plot_fbi_with_areas(max_recalc_fbi, shp_in, extent)

        """
        from fbi_vic_plot_functions import plot_df
        shp_in_cut = shp_in[shp_in['Area_Name']=='East Gippsland']
        plot_df(recalc_file_in['DF_SFC'][28,:,:], shp_in_cut, save_plot = 'df_25mar')
        """
        """
        from fbi_vic_plot_functions import plot_curing
        plot_curing(recalc_file_in['Curing_SFC'][10,:,:], shp_in, save_plot='curing_20mar_morningof')
        """
        
        """Calculate FBI for a region"""
        area_name = 'Central'
        #Update 19/3/24: Use the area template I've created that designates points to a specific FWA.
        map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_FWA_3km.nc")
#        map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_LGA_3km.nc")
        clipped_recalc = max_recalc_fbi.where(map_by_pixel_in['Area_Name']==area_name)
        desig_fbi = np.nanpercentile(clipped_recalc, 90)
        print('The designiated FBI for '+area_name+' is '+str(desig_fbi))

        """Determine the most dominant model"""
        fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv"
        fuel_type_map = recalc_file_in['fuel_type'][10,:,:]
        dom_typ_ = find_dominant_fuel_type_for_a_rating(clipped_recalc, desig_fbi, fuel_type_map, fuel_lut_path)
        print('Dominant model driving rating is '+dom_typ_)
        dom_cod_ = find_dominant_fuel_code_for_a_rating(clipped_recalc, desig_fbi, fuel_type_map, fuel_lut_path, return_table=True)
#        plot_fbi_and_rating_with_fwas(clipped_recalc,max_recalc_rating,shp_in)
        print('Top fuel code is '+str(dom_cod_.index[0]))
        
        
        recalc_file_in.close()
        
