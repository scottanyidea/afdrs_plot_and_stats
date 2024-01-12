#Plots the outputs to the recalculated fire danger grids, compare to official outputs

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import geopandas
from datetime import datetime, timedelta

#Set dates:
year_sel_ = 2024
mon_sel = 1
day_sel = 11

datetime_sel = datetime(year_sel_, mon_sel, day_sel)

forecast_day = 3
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
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/mallee_cases_completed_deltabug/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/mallee_cases_bug_25pccover_7FL/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_fullmalleechanges_grass35/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_grassloadchange/load_35/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")

"""
Find the maximum FBI and FDI at each point: 
Note - there is a need to grab the correct time window, midnight to midnight lOCAL time.    
"""
#start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+day_sel_str+'T13:00:00')
#end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+str(day_sel+1)+'T12:00:00')
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
#start_ind=2
start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
max_recalc_rating = recalc_file_in['rating_1'][start_ind:end_ind,:,:].max(dim='time',keep_attrs=True)

"""Load fire weather area (FWA) shapefile for plotting"""
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")

"""
Plot:
Note at this stage we are plotting day +1 in the official BOM FDI/FBI as the forecast is this
"""
def plot_fbi_and_rating_with_fwas(FBI,rating,areas_shapefile):
    fig, axs = plt.subplots(1,2,figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap_rating = pltcolors.ListedColormap(['white','green','gold','darkorange','red'])
    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5], cmap_rating.N)

    im1 = FBI.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=80., cmap='viridis', add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical')
    cb1.set_label(label='FBI',size=14)
    cb1.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    im2 = rating.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical')
    cb2.set_label(label='Rating',size=14)
    cb2.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI', fontsize=16)
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,147.8,-38.4,-33.8])
    axs[0].set_extent([140.8,145,-37.8,-33.8])
#    axs[0].set_extent([142,144,-36.3,-34])
    axs[1].coastlines()
    axs[1].set_title('Rating',fontsize=16)
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    #axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[1].set_extent([140.8,147.8,-38.4,-33.8])
    axs[1].set_extent([140.8,145,-37.8,-33.8])
#    axs[1].set_extent([142,144,-36.3,-34])        
#    fig.suptitle("Day-ahead forecast for "+str(year_sel_)+mon_sel_str+str(day_sel+1), fontsize=22)
    fig.suptitle("Forecast 14 Jan as of 11th, MH changes", fontsize=24)
#    plt.savefig("fbi_rating_recalc_"+str(year_sel_)+mon_sel_str+day_sel_str+"_day "+str(forecast_day)+".png")
    plt.savefig("Jan14_fc_day3_MHchanges.png")
plot_fbi_and_rating_with_fwas(max_recalc_fbi,max_recalc_rating,shp_in)

#recalc_file_in.close()
