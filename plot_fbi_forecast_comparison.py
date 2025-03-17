#Plots the outputs to the recalculated fire danger grids.
#Designed to take files of different dates to be able to compare two forecasts.
#e.g. lead times of differing number of days

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import geopandas
from datetime import datetime, timedelta
from shapely.geometry import mapping
import cartopy.crs as ccrs
from fbi_vic_plot_functions import plot_and_compare_fbi

#Set first date of forecast:
year_sel_1 = 2024
mon_sel_1 = 12
day_sel_1 = 11
forecast_day = 5

datetime_sel = datetime(year_sel_1, mon_sel_1, day_sel_1)
datetime_fc = datetime_sel + timedelta(days=forecast_day)

#Set second date of forecast:
year_sel_2 = 2024
mon_sel_2 = 12
day_sel_2 = 14
forecast_day2 = 2

datetime_sel_2 = datetime(year_sel_2, mon_sel_2, day_sel_2)
datetime_fc_2 = datetime_sel_2 + timedelta(days=forecast_day)

#set strings here - bit of a mess but helps later!
#Note - we add nothing if we want day+1, bc UTC puts us to next day
mon_sel_str1 = datetime.strftime(datetime_sel, "%m")
day_sel_str1 = datetime.strftime(datetime_sel, "%d")
mon_sel_str_fc1 = datetime.strftime((datetime_sel+timedelta(days=forecast_day-1)), "%m")
day_sel_str_fc1 = datetime.strftime((datetime_sel+timedelta(days=forecast_day-1)), "%d")
day_sel_str_fcplus1_1 = datetime.strftime((datetime_sel+timedelta(days=forecast_day)), "%d")
mon_sel_str_fcplus1_1 = datetime.strftime((datetime_sel+timedelta(days=forecast_day)), "%m")

mon_sel_str2 = datetime.strftime(datetime_sel_2, "%m")
day_sel_str2 = datetime.strftime(datetime_sel_2, "%d")
mon_sel_str_fc2 = datetime.strftime((datetime_sel_2+timedelta(days=forecast_day2-1)), "%m")
day_sel_str_fc2 = datetime.strftime((datetime_sel_2+timedelta(days=forecast_day2-1)), "%d")
day_sel_str_fcplus1_2 = datetime.strftime((datetime_sel_2+timedelta(days=forecast_day2)), "%d")
mon_sel_str_fcplus1_2 = datetime.strftime((datetime_sel_2+timedelta(days=forecast_day2)), "%m")


#load the file:
recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_control/VIC_"+str(year_sel_1)+mon_sel_str1+day_sel_str1+"_recalc.nc")
recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_control/VIC_"+str(year_sel_2)+mon_sel_str2+day_sel_str2+"_recalc.nc")
"""
Find the maximum FBI and FDI at each point: 
Note - there is a need to grab the correct time window, midnight to midnight lOCAL time.    
"""

start_time_ = np.datetime64(str(year_sel_1)+'-'+mon_sel_str_fc1+'-'+day_sel_str_fc1+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_1)+'-'+mon_sel_str_fcplus1_1+'-'+day_sel_str_fcplus1_1+'T12:00:00')
start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
max_recalc_fbi = xr.where(max_recalc_fbi<0, np.nan, max_recalc_fbi)

start_time_ = np.datetime64(str(year_sel_2)+'-'+mon_sel_str_fc2+'-'+day_sel_str_fc2+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_2)+'-'+mon_sel_str_fcplus1_2+'-'+day_sel_str_fcplus1_2+'T12:00:00')
start_ind = np.where(recalc2_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc2_in.time.values==end_time_)[0][0]
max_fbi2 = recalc2_in['index_1'][start_ind:end_ind, :,:].max(dim='time',keep_attrs=True)
max_fbi2 = xr.where(max_fbi2<0, np.nan, max_fbi2)

"""Load fire weather area (FWA) shapefile for plotting"""
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp")

"""
Plot:
Note at this stage we are plotting day +1 in the official BOM FDI/FBI as the forecast is this
"""    
#    extent= [140.8,144.7,-37.0,-33.8]  #Mallee
#    extent= [140.8,144.7,-37.6,-34.8]   #Wimmera
#extent=[140.8,144.7,-38.9,-36.7]   #South West
#    extent = [140.8,145.3,-39,-36.2]   #Custom W and SW (to include Greater Bendigo)
#    extent=[140.8,145.7,-39,-33.8]   #Western half Vic
extent=[140.8,149.9,-39,-33.8]   #All Vic
#extent=[140.8,141.9,-38.6,-37.2]  #Far southwest
plot_and_compare_fbi(max_recalc_fbi, max_fbi2, shp_in, extent)

area_name = 'South West'
area_polygon = shp_in[shp_in['Area_Name']==area_name]
max_recalc_fbi.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
max_recalc_fbi.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
clipped_recalc = max_recalc_fbi.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)

max_fbi2.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
max_fbi2.rio.write_crs("EPSG:4326",inplace=True)
clipped2 = max_fbi2.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
clipped_recalc = xr.where(clipped_recalc<0, np.nan, clipped_recalc)
clipped2 = xr.where(clipped2<0, np.nan, clipped2)
desig_fbi = np.nanpercentile(clipped_recalc, 90)
desig2 = np.nanpercentile(clipped2, 90)
print('The original designiated FBI for '+area_name+' for forecast 1 is '+str(desig_fbi))
print('The designiated FBI for '+area_name+' for forecast 2 is '+str(desig2))



plot_and_compare_fbi(clipped_recalc, clipped2, shp_in, extent)

recalc_file_in.close()
recalc2_in.close()