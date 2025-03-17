#Plots the outputs to the recalculated fire danger grids, compare to official outputs

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import geopandas
from datetime import datetime, timedelta
from shapely.geometry import mapping
import cartopy.crs as ccrs

#Set dates:
year_sel_ = 2025
mon_sel = 1
day_sel = 2

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
recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_control/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_feb_mar23/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_adjusted_kbdi/swvic_testing_nov24/curing90_only/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")

#recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_new_lut/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc_AM.nc")
#recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")

#recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/grass_moisturemod_HC/recalc_files/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_adjusted_kbdi/swvic_testing_nov24/plus30/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_adjusted_kbdi/swvic_testing_nov24/plus40_curing90/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
recalc2_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_curing_adj/d10_add20/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
"""
Find the maximum FBI and FDI at each point: 
Note - there is a need to grab the correct time window, midnight to midnight lOCAL time.    
"""

#start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+day_sel_str+'T13:00:00')
#end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+str(day_sel+1)+'T12:00:00')
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
#start_ind=2
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
max_recalc_ffdi = recalc_file_in['GFDI_SFC'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
max_recalc_ffdi = xr.where(max_recalc_ffdi<0, np.nan, max_recalc_ffdi)
max_ffdi2 = recalc2_in['GFDI_SFC'][start_ind:end_ind, :,:].max(dim='time',keep_attrs=True)
max_ffdi2 = xr.where(max_ffdi2<0, np.nan, max_ffdi2)

"""Load fire weather area (FWA) shapefile for plotting"""
#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp")
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90209_VIC_Boundary_SHP_CFA\PID90209_VIC_Boundary_SHP_CFA.shp")

"""
Plot:
Note at this stage we are plotting day +1 in the official BOM FDI/FBI as the forecast is this
"""
def plot_and_compare_ffdi(max_ffdi1, max_ffdi2, areas_shapefile, fc_day):
    fig, axs = plt.subplots(1,3,figsize=(14,4), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap_rating = pltcolors.ListedColormap(['green','blue','wheat','yellow','gold','darkorange','darkred'])
#    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5], cmap_rating.N)
    norm = pltcolors.BoundaryNorm([0,12,24,35,50,75,100,200], cmap_rating.N)

    im1= max_ffdi1.plot(ax=axs[0], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.035)
#    cb1.set_label(label='FBI',size=14)
    cb1.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    im2 = max_ffdi2.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical', fraction=0.035)
#    cb2.set_label(label='Maximum FBI in day',size=14)
    cb2.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('Original', size=16)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.0,-33.8])  #Mallee
#    axs[0].set_extent([140.8,144.7,-37.6,-33.8])
#    axs[0].set_extent([142.8,145.7,-38.8,-37.0])
#    axs[0].set_extent([140.8,144.7,-37.6,-34.8])   #Wimmera
#    axs[0].set_extent([140.8,144.7,-38.9,-36.7])   #South West
#    axs[0].set_extent([140.8,145.3,-39,-36.2])   #Custom W and SW (to include Greater Bendigo)
#    axs[0].set_extent([140.8,145.7,-39,-33.8])   #Western half Vic
#    axs[0].set_extent([140.8,149.9,-39,-33.8])   #All Vic
    axs[0].set_extent([145., 148, -39.2, -36.5])   #Gippsland


    axs[1].coastlines()
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_title('Changed', size=16)
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[1].set_extent([140.8,144.7,-37.0,-33.8])  #Mallee
#    axs[1].set_extent([140.8,144.7,-37.6,-33.8])
#    axs[1].set_extent([140.8,144.7,-37.6,-34.8])   #Wimmera
#    axs[1].set_extent([140.8,144.7,-38.9,-36.7])   #South West
#    axs[1].set_extent([140.8,145.5,-39,-36.2])   #Custom W and SW (to include Greater Bendigo)
#    axs[1].set_extent([140.8,145.7,-39,-33.8])   #Western half Vic
#    axs[1].set_extent([140.8,149.9,-39,-33.8])   #All Vic
    axs[1].set_extent([145., 148, -39.2, -36.5])   #Gippsland

    fig.suptitle("FBI forecast init "+str(year_sel_)+mon_sel_str+day_sel_str+", day "+str(fc_day), fontsize=20)


    
    """Calc difference and plot"""
    max_ffdi2 = max_ffdi2.interp(latitude = max_ffdi1.latitude, longitude=max_ffdi1.longitude, method='nearest')
    difference_ffdi = max_ffdi2 - max_ffdi1
    
    im3 = difference_ffdi.plot(ax=axs[2], transform=ccrs.PlateCarree(),vmin=-30., vmax=30., cmap='RdYlGn_r', add_colorbar=False)
    cb3 = plt.colorbar(im3, orientation='vertical', fraction=0.035)
    cb3.ax.tick_params(labelsize=12)
    areas_shapefile.plot(ax=axs[2], facecolor="none")
    axs[2].coastlines()
    axs[2].set_title('Diff changed - original', size=16)
#    axs[2].set_extent([140.8,144.7,-37.0,-33.8]) #Mallee
#    axs[2].set_extent([140.8,144.7,-37.6,-33.8])
#    axs[2].set_extent([140.8,144.7,-37.6,-34.8])   #Wimmera
#    axs[2].set_extent([140.8,144.7,-38.9,-36.7])   #South West
#    axs[2].set_extent([140.8,145.5,-39,-36.2])   #Custom W and SW (to include Greater Bendigo)
#    axs[2].set_extent([140.8,145.7,-39,-33.8])   #Western half Vic
#    axs[2].set_extent([140.8,149.9,-39,-33.8])   #Western half Vic
    axs[2].set_extent([145., 148, -39.2, -36.5])   #Gippsland

#    plt.savefig("fbi_recalc_"+str(year_sel_)+mon_sel_str+day_sel_str+"_kbdi40_day "+str(forecast_day)+"_.png")
    
plot_and_compare_ffdi(max_recalc_ffdi, max_ffdi2, shp_in, forecast_day)

area_name = '10'
area_polygon = shp_in[shp_in['Area_Name']==area_name]
max_recalc_ffdi.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
max_recalc_ffdi.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
clipped_recalc = max_recalc_ffdi.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)

max_ffdi2.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
max_ffdi2.rio.write_crs("EPSG:4326",inplace=True)
clipped2 = max_ffdi2.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
clipped_recalc = xr.where(clipped_recalc<0, np.nan, clipped_recalc)
clipped2 = xr.where(clipped2<0, np.nan, clipped2)
desig_fbi = np.nanpercentile(clipped_recalc, 90)
desig2 = np.nanpercentile(clipped2, 90)
print('The original designiated FBI for '+area_name+' is '+str(desig_fbi))
print('The designiated FBI for '+area_name+' with increased winds is '+str(desig2))



#plot_and_compare_fbi(clipped_recalc, clipped2, shp_in, forecast_day)

recalc_file_in.close()
recalc2_in.close()