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
import seaborn

def plot_fbi_and_rating_with_fwas(FBI,rating,areas_shapefile):
    fig, axs = plt.subplots(1,2,figsize=(14,6), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap_rating = pltcolors.ListedColormap(['white','green','gold','darkorange','darkred'])
#    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5], cmap_rating.N)
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,200], cmap_rating.N)

    im1 = FBI.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=100., cmap='viridis', add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.032)
    cb1.set_label(label='FBI',size=14)
    cb1.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
#    im2 = rating.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    im2 = FBI.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical', fraction=0.032)
    cb2.set_label(label='Maximum FBI in day',size=16)
#    cb2.set_label(label='Maximum FBI in day',size=14)
    cb2.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI', fontsize=18)
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,147.8,-38.4,-33.8])
#    axs[0].set_extent([140.8,145,-37.8,-33.8])
    axs[0].set_extent([140.8,144.7,-37.6,-34.8])   #Wimmera
#    axs[0].set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
#    axs[0].set_extent([140.8,144.7,-38.9,-36.7])   #South West
#    axs[0].set_extent([140.8,150,-39.3,-33.8])  #most of Vic


#    axs[0].set_extent([142,144,-36.3,-34])
    axs[1].coastlines()
    axs[1].set_title('FBI',fontsize=18)
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    #axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[1].set_extent([140.8,147.8,-38.4,-33.8])
#    axs[1].set_extent([140.8,145,-37.8,-33.8])
#    axs[1].set_extent([140.8,144.7,-37.6,-34.8])   #Wimmera
#    axs[1].set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
    axs[1].set_extent([140.8,144.7,-38.9,-36.7])   #South West
#    axs[1].set_extent([140.8,150,-39.3,-33.8])  #most of Vic
#    axs[1].set_extent([142,144,-36.3,-34])        
#    fig.suptitle("Day-ahead forecast for "+str(year_sel_)+mon_sel_str+str(day_sel+1), fontsize=22)
#    fig.suptitle("18 Mar 2023", fontsize=24)
#    plt.savefig("fbi_rating_recalc_"+str(year_sel_)+mon_sel_str+day_sel_str+".png")
    plt.savefig("Feb28_fc_onday_df95_wimmera.png")
    
    axs[1].text(142.2, -37.88, 'South West', fontsize=12 )
    axs[1].text(144, -37.8, 'Central', fontsize=12)
    axs[1].text(141.8, -36.9, 'Wimmera', fontsize=12)
    axs[1].text(143.7, -37.05, 'N. Central', fontsize=12)
        
    
    
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
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/fixed_df/df_95/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
#        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/IDZ10133_AUS_AFDRS_max_fbi_prelim_SFC.nc")
        """
        Find the maximum FBI and FDI at each point: 
        Note - there is a need to grab the correct time window, midnight to midnight LOCAL time.    
        """
        #start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+day_sel_str+'T13:00:00')
        #end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+str(day_sel+1)+'T12:00:00')
        start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
        end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
#        start_ind=3
        start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
        end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
        max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
#        max_recalc_fbi = recalc_file_in['max_fbi_prelim'][1,:,:]
#        max_recalc_fbi = recalc_file_in['FDI_SFC'][:,:,start_ind:end_ind].max(dim='time',keep_attrs=True)
        max_recalc_rating = recalc_file_in['rating_1'][start_ind:end_ind,:,:].max(dim='time',keep_attrs=True)
#        max_recalc_rating = rating_file_in['max_fdr_prelim'][1,:,:]

        """Load fire weather area (FWA) shapefile for plotting"""
        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
        #shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp")
#        shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90309_VIC_Boundary_SHP_ICC\PID90309_VIC_Boundary_SHP_ICC.shp")

        plot_fbi_and_rating_with_fwas(max_recalc_fbi,max_recalc_rating,shp_in)

        """        
        from fbi_vic_plot_functions import plot_df
        plot_df(recalc_file_in['DF_SFC'][6,:,:], shp_in)
        """
        
        """Calculate FBI for a region"""
        area_name = 'South West'
        area_polygon = shp_in[shp_in['Area_Name']==area_name]
        max_recalc_fbi.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
        max_recalc_fbi.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
        clipped_recalc = max_recalc_fbi.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        desig_fbi = np.nanpercentile(clipped_recalc, 90)
        print('The designiated FBI for '+area_name+' is '+str(desig_fbi))
        
        """Determine the most dominant model"""
        fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231012043244.csv"
        fuel_type_map = recalc_file_in['fuel_type'][10,:,:]
        dom_typ_ = find_dominant_fuel_type_for_a_rating(clipped_recalc, desig_fbi, fuel_type_map, fuel_lut_path)
        print('Dominant model driving rating is '+dom_typ_)
        dom_cod_ = find_dominant_fuel_code_for_a_rating(clipped_recalc, desig_fbi, fuel_type_map, fuel_lut_path, return_table=True)
        plot_fbi_and_rating_with_fwas(clipped_recalc,max_recalc_rating,shp_in)
        print('Top fuel code is '+str(dom_cod_.index[0]))
        """Distribution of grass fuel loads on grass pixels"""
        #Clip first to grass pixels >90th percentile
        clipped_recalc_pct = clipped_recalc.where((clipped_recalc >= desig_fbi))
        fuel_type_pct = xr.where(~np.isnan(clipped_recalc_pct), fuel_type_map, np.nan)  #fuel type map is now restricted to that nth percentile
        clipped_grass_pct = clipped_recalc_pct.where(fuel_type_pct.isin([3004,3016,3020,3042,3044,3046,3062,3064]))
        
        #Now get the data.
        grass_curing_pct = recalc_file_in['Curing_SFC'][3,:,:].where(~np.isnan(clipped_grass_pct)).to_dataframe().dropna(subset='Curing_SFC')
        grass_load_pct = recalc_file_in['GrassFuelLoad_SFC'][3,:,:].where(~np.isnan(clipped_grass_pct)).to_dataframe().dropna(subset='GrassFuelLoad_SFC')
        colors_ = seaborn.color_palette('bright')
        
        plt.figure()
        seaborn.histplot(grass_curing_pct['Curing_SFC'], label='Curing', bins=np.arange(90,101,1), color=colors_[0])
        plt.figure()
        seaborn.histplot(grass_load_pct['GrassFuelLoad_SFC'], label='Curing', bins=np.arange(1,6,0.5), color=colors_[0])
        
        
#        recalc_file_in.close()
