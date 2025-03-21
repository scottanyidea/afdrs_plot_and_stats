#Plots the outputs to the recalculated fire danger grids, compare to official outputs

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geopandas
from datetime import datetime, timedelta
from shapely.geometry import mapping
from fbi_vic_plot_functions import plot_fbi_and_fdi, plot_fbi_ffdi_gfdi_ratings

#Set dates:
year_sel_ = 2025
mon_sel = 3
day_sel = 16

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
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_windonhour/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc_AM.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/fixed_df/df_95/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/grass_curing_20240226/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")

"""
Find the maximum FBI and FDI at each point: 
Note - there is a need to grab the correct time window, midnight to midnight lOCAL time.    
"""
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
if forecast_day==0:
    start_ind=1
else:
    start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
max_recalc_fdi = recalc_file_in['FDI_SFC'][:,:,start_ind:end_ind].max(dim='time', keep_attrs=True)
max_recalc_gfdi = recalc_file_in['GFDI_SFC'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
max_recalc_ffdi = recalc_file_in['FFDI_SFC'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)

"""Load fire weather area (FWA) shapefile for plotting"""
#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp")

"""Load also the template file for clipping"""
#area_template = xr.open_dataset("C://Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_FWA_3km.nc")
area_template = xr.open_dataset("C://Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_LGA_3km.nc")

"""
Plot:
Note at this stage we are plotting day +1 in the official BOM FDI/FBI as the forecast is this
"""
area_name = 'Towong'
shp_in_sel = shp_in[shp_in['Area_Name']==area_name]

#extent = [140.8,150.2,-39,-33.8] #all Vic
#extent = [140.8,145.7,-39,-33.8] #western half Vic
#extent = [143.2,145.3,-38,-36.3] #Melbourne and a little west and north...
#extent = [143.7,144.7,-37.4,-36.7] #Mt Alexander LGA
#extent = [147.0,150,-38.1,-36.4]   #East Gippsland
#extent = [146.0,148.0,-38.8,-36.9] #Central Gippsland
#extent = [140.8,144.7,-38.9,-36.7]   #South West
extent = [145.5, 148.5, -36.9, -35.5]  #North East
plot_fbi_and_fdi(max_recalc_fbi, max_recalc_fdi, shp_in,extent, save_plot='./test.png')
#plot_fbi_and_fdi(max_recalc_fbi, max_recalc_fdi, shp_in,extent)

"""Calculate FBI for a region"""

clipped_recalc = xr.where(area_template['Area_Name']==area_name, max_recalc_fbi, np.nan)
plot_fbi_and_fdi(clipped_recalc, max_recalc_fbi, shp_in,extent, save_plot='./test.png')
desig_fbi = np.nanpercentile(clipped_recalc, 90)
print('The designiated FBI for '+area_name+' is '+str(desig_fbi))


clipped_recalcffdi = xr.where(area_template['Area_Name']==area_name, max_recalc_ffdi, np.nan)
desig_ffdi = np.nanpercentile(clipped_recalcffdi, 90)
print('The designiated FFDI for '+area_name+' is '+str(desig_ffdi))


clipped_recalcgfdi = xr.where(area_template['Area_Name']==area_name, max_recalc_gfdi, np.nan)
desig_gfdi = np.nanpercentile(clipped_recalcgfdi, 90)
print('The designiated GFDI for '+area_name+' is '+str(desig_gfdi))


clipped_recalcfdi = xr.where(area_template['Area_Name']==area_name, max_recalc_fdi, np.nan)
desig_fdi = np.nanpercentile(clipped_recalcfdi, 90)
print('The designiated FDI for '+area_name+' is '+str(desig_fdi))

#plot_fbi_ffdi_gfdi_ratings(max_recalc_fbi, max_recalc_ffdi, max_recalc_gfdi, shp_in, extent, save_plot = 'mar25_eveningprior_fbi_fdis.png')

recalc_file_in.close()