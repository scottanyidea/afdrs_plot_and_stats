#Plots the outputs to the recalculated fire danger grids, compare to official outputs.
#Do so for selected fuel types.

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geopandas
from datetime import datetime, timedelta

#Set dates:
year_sel_ = 2024
mon_sel = 1
day_sel = 12

datetime_sel = datetime(year_sel_, mon_sel, day_sel)

forecast_day = 2
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
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/mallee_cases_to_spinifex/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_grassloadchange/load_35/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
official_fbi_in = xr.open_dataset("M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/2023-2024/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str+"/IDZ10137_AUS_AFDRS_max_fbi_SFC.nc.gz")

"""Make a mask for the fuel types we want."""
fuel_types_ = [3024,3048]
mask_fuel_types = np.isin(recalc_file_in['fuel_type'].values[0,:,:], fuel_types_)

"""
Find the maximum FBI and FDI at each point: 
Note - there is a need to grab the correct time window, midnight to midnight lOCAL time.    
"""
#start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+day_sel_str+'T13:00:00')
#end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+str(day_sel+1)+'T12:00:00')
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]

max_recalc_value = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
max_recalc_value = max_recalc_value.where(mask_fuel_types)
max_orig_value =  official_fbi_in['MaxFBI_SFC'][forecast_day,:,:]
max_orig_value = max_orig_value.where(mask_fuel_types)

"""Load fire weather area (FWA) shapefile for plotting"""
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")

"""
Plot:
Note at this stage we are plotting day +1 in the official BOM FDI/FBI as the forecast is this
"""

fig, axs = plt.subplots(1,3,figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})

max_recalc_value.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=80., cmap='viridis')
shp_in.plot(ax=axs[0], facecolor="none")
max_orig_value.plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=0., vmax=80., cmap='viridis')
shp_in.plot(ax=axs[1], facecolor="none")
axs[0].coastlines()
axs[0].set_title('Recalc FBI')
axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
axs[0].set_extent([140.8,144.7,-37.0,-33.8])
axs[1].coastlines()
axs[1].set_title('FBI from BOM forecast')
axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
axs[1].set_extent([140.8,144.7,-37.0,-33.8])
fig.suptitle("forecast init "+str(year_sel_)+mon_sel_str+day_sel_str+", day "+str(forecast_day), fontsize=20)

"""Calc difference and plot"""
difference_fbi = max_recalc_value - max_orig_value

difference_fbi.plot(ax=axs[2], transform=ccrs.PlateCarree(),vmin=-10., vmax=10., cmap='RdYlGn_r')
shp_in.plot(ax=axs[2], facecolor="none")
axs[2].coastlines()
axs[2].set_title('Diff recalc - BOM')
axs[2].gridlines(draw_labels=False)
axs[2].set_extent([140.8,144.7,-37.0,-33.8])

#plt.savefig("fbi_only_recalc_vs_bom_"+str(year_sel_)+mon_sel_str+day_sel_str+"_day "+str(forecast_day)+"_spinifex_pt.png")

recalc_file_in.close()
official_fbi_in.close()