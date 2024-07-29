"""
#Plots FBI as points within the area of a fire boundary. Commented options
to instead plot fuel type.

"""
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import geopandas
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import mapping, Point

def plot_fbi_points_in_fire(FBI,areas_shapefile):
    fig, axs = plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.PlateCarree()})
    colors=['blue', 'purple', 'lightgreen']
    
    unique_models = np.unique(FBI['FBM'])
    k=0
    areas_shapefile.plot(ax=axs, facecolor="navajowhite")
    for model_type in unique_models:
        table_sub = FBI[FBI['FBM']==model_type]
        im1 = table_sub.plot(ax=axs, transform=ccrs.PlateCarree(),marker='o', color=colors[k], markersize=50, label=model_type)
#        for x, y, indx in zip(table_sub.longitude, table_sub.latitude, table_sub.index_1):
        for x, y, indx in zip(table_sub.longitude, table_sub.latitude, table_sub.fuel_type):
            axs.text(x, y, int(indx), fontsize=18)
        
        k=k+1
    axs.legend(fontsize=18)

    plt.savefig('briagolong_1oct_fueltypes.png')
    
    
if __name__=="__main__":
    #Set dates:
        year_sel_ = 2023
        mon_sel = 9
        day_sel = 30

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
        recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_control/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
        """
        Find the maximum FBI and FDI at each point: 
        Note - there is a need to grab the correct time window, midnight to midnight LOCAL time.    
        """
        start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
        end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
#        start_ind=12
        start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
        end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
        max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)
#        max_recalc_fbi = recalc_file_in['max_fbi_prelim'][1,:,:]
#        max_recalc_fbi = recalc_file_in['FDI_SFC'][:,:,start_ind:end_ind].max(dim='time',keep_attrs=True)
        max_recalc_rating = recalc_file_in['rating_1'][start_ind:end_ind,:,:].max(dim='time',keep_attrs=True)
#        max_recalc_rating = rating_file_in['max_fdr_prelim'][1,:,:]
        fuel_types = recalc_file_in['fuel_type'][12,:,:]
        
        """Load shapefile for plotting"""
#        shp_in = geopandas.read_file("C:/Users/clark/analysis1/Case_studies/2024_02_22/shapefiles/Obs_20240222_1959/Obs_area.shp")
#        shp_in = geopandas.read_file("C:/Users/clark/analysis1/Case_studies/2024_02_22/shapefiles/Obs_20240223_1959/Obs_area.shp")
#        shp_in = geopandas.read_file("C:/Users/clark/analysis1/Case_studies/2024_03_26/Obs_20240328_0859/Obs_area.shp")
        shp_in = geopandas.read_file("C:/Users/clark/analysis1/Case_studies/2023_10_02/Obs_20231002_1159/Obs_area.shp")
#        shp_in_subset = shp_in[shp_in['DSE_ID']==999213]
        shp_in_subset = shp_in[shp_in['CFA_ID']==1961458]
        #The polygon gives weird coordinates. Turns out it's just in a projected coordinate system, 
        #gotta change it to a geographic reference system. So convert to 4326 (WGS 84)
        shp_in_subset = shp_in_subset.to_crs(4326)
        max_recalc_fbi.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
        max_recalc_fbi.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
        #Clip to the fire area:
        clipped_recalc = max_recalc_fbi.rio.clip(shp_in_subset.geometry.apply(mapping), shp_in_subset.crs, drop=False)

        #merge on fuel type codes:
        clipped_recalc_ft = xr.merge([clipped_recalc, fuel_types])
        #convert to pd dataframe:
        fbi_table_ = clipped_recalc_ft.to_dataframe().dropna(subset='index_1')
        #load fuel lut for mapping on fuel categories:
        path_to_fuel_lut_ = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv"
        fuel_lut_ = pd.read_csv(path_to_fuel_lut_)
        fuel_FBM_dict = pd.Series(fuel_lut_.FBM.values,index=fuel_lut_.FTno_State).to_dict()
        fbi_table_['FBM'] = fbi_table_['fuel_type'].map(fuel_FBM_dict)
        fbi_table_ = fbi_table_.drop(columns=['band','spatial_ref', 'time']).reset_index()
        #map lat and lon as shapely objects in geopandas:
        geometry = [Point(xy) for xy in zip(fbi_table_['longitude'], fbi_table_['latitude'])]
        fbi_gdf = geopandas.GeoDataFrame(fbi_table_, geometry=geometry)
        
        plot_fbi_points_in_fire(fbi_gdf, shp_in_subset)
        
        recalc_file_in.close()
