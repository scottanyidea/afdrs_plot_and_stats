#Replaces one or multiple fuel types in the historical (datacube) with a recalculated
#version that has been modified in some way, e.g. perturbed, updated or has bug fixes.
#Then calculates the new FBI for a specific region (here it's done by FWA but can be
#replaced with LGAs, etc)

#This code assumes the recalculated, modified data has NaNs where it has not been modified.
#Presumably, if it hasn't been modified and it's not NaNs, you're just replacing one value with an
#identical one...

#Edit Jan 24: To take multiple single model recalcs and apply all of them to the original data.
#Currently a work in progress...

import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
from datetime import datetime
import time


def replace_fuel_calc_rating(original_path, newdata_paths, date_list, area_file_path, area_name, fuel_lut_pth_orig, fuel_lut_pth_new=None):
    """
    Over a fire behaviour index dataset (such as recalculated VicGrids datacube), replace one or multiple fuel types with another
    calculated version that has had changes to fuel parameters or the model used.

    This assumes daily files with the format "VIC_<date>_recalc.nc"

    Parameters
    ----------
    original_path : String for the path to a folder containing the official or original daily data files (e.g. VicGrids or a full recalc thereof). 
    
    newdata_paths : String or list of strings for the path(s) to the folders containing the files including the model or parameter changes.
    
    date_list : List of dates to run the calculation over. If the data is unavailable for a date in this list, it will be skipped.
    
    area_file_path : Path to a shapefile (file, not folder!) that defines the regions we want to designate a rating for.
    
    area_name : Name of the region within above shapefile that we want to designate FBIs and ratings for..

    Returns
    -------
    rating_table : Pandas dataframe that contains daily FBI, rating for the original daily data and the data with the changes implemented.

    """
    dates_used = []
    fbi_orig = []
    fbi_recalc = []
    rating_orig = []
    rating_recalc = []
    orig_dom_type = []
    recalc_dom_type = []
    for dt in date_list:
        time_start = time.time()
        print('starting '+str(dt))
        try:
            #First get the files.
            date_str = dt.strftime("%Y%m%d")
            official_file_in = xr.open_dataset(original_path+'VIC_'+date_str+'_recalc.nc')
            
            #Find maximum FBI along the day
            official_max = official_file_in['index_1'].max(dim='time',skipna=True, keep_attrs=True)
        
            #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
            shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp", crs='ESPG:4326')
            area_polygon = shp_in[shp_in['Area_Name']==area_name]

            official_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
            official_max.rio.write_crs("EPSG:4326",inplace=True)
            clipped_orig = official_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
            
            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_orig, 90)
            desig_rating = rating_calc(desig_fbi)
            fbi_orig.append(desig_fbi)
            rating_orig.append(desig_rating)
            
            #Find the fuel type that is most dominant in the 90th percentile.
            fuel_type_map_fbi = official_file_in['fuel_type']
            dom_typ = find_dominant_fuel_type_for_a_rating(clipped_orig, desig_fbi, fuel_type_map_fbi,fuel_lut_pth_orig)
            orig_dom_type.append(dom_typ)
            
            #Do the same to the recalculated data, iteratively
            clipped_replaced = xr.DataArray(coords=(clipped_orig['latitude'], clipped_orig['longitude']))
            for recalcs in newdata_paths:
                recalc_file_in = xr.open_dataset(recalcs+"VIC_"+date_str+"_recalc.nc")
                recalc_max = recalc_file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)
                recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) 
                recalc_max.rio.write_crs("EPSG:4326",inplace=True)
                clipped_recalc = recalc_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                clipped_replaced = clipped_replaced.fillna(clipped_recalc)
                recalc_file_in.close()
            
            clipped_replaced = clipped_replaced.fillna(clipped_orig)
            clipped_replaced.name = 'index_1'
            recalc_fbi = np.nanpercentile(clipped_replaced, 90)
            recalc_rating = rating_calc(recalc_fbi)
            fbi_recalc.append(recalc_fbi)
            rating_recalc.append(recalc_rating)
            
            #Set "new" fuel lut to the old if it hasn't been set:
            if fuel_lut_pth_new is None:
                fuel_lut_pth_new = fuel_lut_pth_orig
            
            #Find most dominant fuel type for the recalculated data.
            dom_typ_recalc = find_dominant_fuel_type_for_a_rating(clipped_replaced, recalc_fbi, fuel_type_map_fbi, fuel_lut_pth_new)
            recalc_dom_type.append(dom_typ_recalc)
            
            #If we get to this point, grab the date.
            dates_used.append(dt)
            
            official_file_in.close()
        except FileNotFoundError:
            print(date_str+" not found. Skip to next")
            pass
        finally:
            time_end = time.time()
            print('Time for this iteration: '+str(time_end - time_start))
    rating_table = pd.DataFrame(list(zip(dates_used, fbi_orig, rating_orig, orig_dom_type, fbi_recalc, rating_recalc, recalc_dom_type)), columns=['Date', 'Original FBI','Original rating', 'Original Dominant FT', 'Changed FBI','Changed rating', 'Changed dominant FT'])
    
    return rating_table
 
def rating_calc(fbi):
    if fbi < 12:
        rating = "0"
    else:
        if fbi < 24:
            rating = "1"
        else:
            if fbi < 50:
                rating = "2"
            else:
                if fbi <100:
                    rating = "3"
                else:
                    rating = "4"
    return rating

def find_dominant_fuel_type_for_a_rating(fbi_arr, rating_val, fuel_type_map, fuel_lut_path):
    #This assumes the fbi_arrat
    
    #Mask fuel type map to be same as the FBI map:
    fuel_type_map_clipped = xr.where(fbi_arr, fuel_type_map[0,:,:], np.nan)
    fuel_type_map_clipped.name = 'fuel_type'
    
    #Merge FBI with fuel types, and mask to only those pixels above 90th percentile
    merged_fbi_ft = xr.merge([fbi_arr, fuel_type_map_clipped])
    merged_fbi_ft = merged_fbi_ft.where((merged_fbi_ft['index_1'] >= rating_val))    #get only those values above say 90th percentile
    top_pixels_table = merged_fbi_ft.to_dataframe()
    top_pixels_table.dropna(axis=0, inplace=True)

    #Load the fuel lut to match fuel types to the codes and pixels:
    fuel_lut = pd.read_csv(fuel_lut_path)
    fuel_FBM_dict = pd.Series(fuel_lut.FBM.values,index=fuel_lut.FTno_State).to_dict()
    top_pixels_table['FBM'] = top_pixels_table['fuel_type'].map(fuel_FBM_dict)
    
    #If the highest ranked fuel model has less than half the points, return "none" as
    #we don't consider it dominant. Else, return the name of the model.
    #OR: if we have a small region, sometimes all the pixels have a zero FBI and it 
    #somehow messes up the grater than or equal to function even if threshold is also zero.
    #In these cases also set "None".
    if (len(top_pixels_table)==0) or (top_pixels_table.FBM.value_counts().iloc[0]/top_pixels_table.FBM.value_counts().sum() < 0.5):
        topmodel = 'None'
    else:
        topmodel = top_pixels_table.FBM.value_counts().index[0]
        
    return topmodel  #ie. return the NAME of the top fuel type

if __name__=="__main__":
    dc_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/v2024.1b7/full_recalc_jan_24/recalc_files/'
    recalc_path = ['C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/mallee_only_add_ltldesert_jan24/recalc_files/']
    #               'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/recalc_grass_35/recalc_files/']

    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp"

    path_to_fuel_lut_orig = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231012043244.csv"
    path_to_fuel_lut_recalc = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231012043244_pt_ltldesmallee.csv"

    area_name = 'Wimmera'

    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,30), freq='D')
    #dates_ = pd.date_range(datetime(2020,4,4), datetime(2020,4,4), freq='D')

    fbi_and_rating = replace_fuel_calc_rating(dc_path, recalc_path, dates_, shp_path, area_name, path_to_fuel_lut_orig, path_to_fuel_lut_recalc)
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_jan24/wimmera_ltldes_mal/wimmera_historical_fbi_rating_ltldesertmal.csv")