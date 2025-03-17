#Replaces one or multiple fuel types in the historical (datacube) with a recalculated
#version that has been modified in some way, e.g. perturbed, updated fuel table or has bug fixes.
#Then calculates the new FBI for a specific region (here it's done by FWA but can be
#replaced with LGAs, etc)

#This code assumes the recalculated, modified data has NaNs where it has not been modified.
#Presumably, if it hasn't been modified and it's not NaNs, you're just replacing one value with an
#identical one...

#Edit Jan 24: To take multiple single model recalcs and apply all of them to the original data.

#This assumes the file "datacube_region_designate_rating.py"
#is in the same directory as this.

import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import multiprocessing as mp

def replace_fuel_calc_rating(original_path, newdata_paths, date_in, area_mask, fuel_lut_pth_orig, fuel_lut_pth_new=None, plot_comp=False):
    from datacube_region_designate_rating import find_dominant_fuel_type_for_a_rating, rating_calc
    """
    For a date, replace one or multiple fuel types with another
    calculated version that has had changes to fuel parameters or the model used.

    This assumes daily files with the format "VIC_<date>_recalc.nc"

    Parameters
    ----------
    original_path : String for the path to a folder containing the official or original daily data files (e.g. VicGrids or a full recalc thereof). 
    
    newdata_paths : String or list of strings for the path(s) to the folders containing the files including the model or parameter changes.
    
    date_in : Dates to run the calculation for. If the data is unavailable, it will be skipped.
    
    area_polygon : A geopandas shape (polygon) that defines the region over which we want to calculate the rating.
    
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
    try:
            #First get the files.
            date_str = date_in.strftime("%Y%m%d")
            official_file_in = xr.open_dataset(original_path+'VIC_'+date_str+'_recalc.nc')
            
            #Find maximum FBI along the day
            official_max = official_file_in['index_1'].max(dim='time',skipna=True, keep_attrs=True)        
            
            #Area mask to the desired region:
            clipped_orig = official_max.where(~area_mask.isnull())
            
            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_orig, 90)
            desig_rating = rating_calc(desig_fbi)
            
            #Find the fuel type that is most dominant in the 90th percentile.
            #Out of that top 10%, if there is a fuel type consists of more than
            #half of those pixels the model is considered "dominant".
            fuel_type_map_fbi = official_file_in['fuel_type']
            orig_dom_type = find_dominant_fuel_type_for_a_rating(clipped_orig, desig_fbi, fuel_type_map_fbi,fuel_lut_pth_orig)
            
            #Do the same to the recalculated data, iteratively
            clipped_replaced = xr.DataArray(coords=(clipped_orig['latitude'], clipped_orig['longitude']))
            for recalcs in newdata_paths:
                recalc_file_in = xr.open_dataset(recalcs+"VIC_"+date_str+"_recalc.nc")
                recalc_max = recalc_file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)
                #As of AFDRS version 2024.5.0 - invalid values return fill values not nans. 
                #So tidy this.
                recalc_max = xr.where(recalc_max<0, np.nan, recalc_max)
                clipped_recalc = recalc_max.where(~area_mask.isnull())
                clipped_replaced = clipped_replaced.fillna(clipped_recalc)
                recalc_file_in.close()
            
            clipped_replaced = clipped_replaced.fillna(clipped_orig)
            clipped_replaced.name = 'index_1'
            recalc_fbi = np.nanpercentile(clipped_replaced, 90)
            recalc_rating = rating_calc(recalc_fbi)
            
            
            #Set "new" fuel lut to the old if it hasn't been set:
            if fuel_lut_pth_new is None:
                fuel_lut_pth_new = fuel_lut_pth_orig
            
            #Find most dominant fuel type for the recalculated data.
            dom_typ_recalc = find_dominant_fuel_type_for_a_rating(clipped_replaced, recalc_fbi, fuel_type_map_fbi, fuel_lut_pth_new)

            
            
            official_file_in.close()
    except FileNotFoundError:
            print(date_str+" not found.")
            

    outputs_ = date_in, desig_fbi, desig_rating, orig_dom_type, recalc_fbi, recalc_rating, dom_typ_recalc
    
    return outputs_

if __name__=="__main__":
    #Paths to the datacube and the replacement recalc(s)
    dc_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/'
    #recalc_path can be a list of multiple paths
    recalc_path = ['C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/forest_tif_changes_aug24/recalc_files/']

#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp"
    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp"    

    #Load fuel luts - this is to take into account changes between models (e.g. mallee heath to forest, etc)
    path_to_fuel_lut_orig = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv"
    path_to_fuel_lut_recalc = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv"
      
    
    area_name = 'Wellington'
    #Get the regional template grid for defining each area:
    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_LGA_1500m.nc")
    map_by_pixel = map_by_pixel_in['Area_Name'].where(map_by_pixel_in['Area_Name']==area_name)

    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,6,1), freq='D')
    #dates_ = pd.date_range(datetime(2021,12,19), datetime(2022,1,4), freq='D')

    #Get a list of the dates actually in the range by checking all the daily files are there.
    dates_used=[]
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(dc_path+'VIC_'+date_str+'_recalc.nc').is_file():
                dates_used.append(dt)
    replace_fuel_calc_rating(dc_path, recalc_path, dates_used[0], map_by_pixel, path_to_fuel_lut_orig, path_to_fuel_lut_recalc)
    
    #Use multiprocessing to go through each daily grid in the datacube.
    #Only really offers a speed advantage if all the data is local
    
    pool = mp.Pool(12)
    start_time = time.time()
    results_pool = [pool.apply_async(replace_fuel_calc_rating, args=(dc_path, recalc_path, dt, map_by_pixel, path_to_fuel_lut_orig, path_to_fuel_lut_recalc)) for dt in dates_used]    
    pool.close()
    pool.join()
    results_list_ = [r.get() for r in results_pool]
    end_time = time.time()
    print("Time taken: "+str(round(end_time-start_time, 3)))
    
    fbi_and_rating = pd.DataFrame(results_list_, columns=['Date', 'Original FBI', 'Original rating', 'Original Dominant FT', 'Changed FBI','Changed rating', 'Changed dominant FT'])
#    fbi_and_rating = replace_fuel_calc_rating(dc_path, recalc_path, dates_used[1], area_in, path_to_fuel_lut_orig, path_to_fuel_lut_recalc)
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_jul24/changes/lga/"+area_name+"_dc_fbi_rating_foresttif.csv")
    