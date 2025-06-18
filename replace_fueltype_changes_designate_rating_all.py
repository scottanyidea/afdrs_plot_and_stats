#Replaces one or multiple fuel types in the historical (datacube) with a recalculated
#version that has been modified in some way, e.g. perturbed, updated or has bug fixes.
#Then calculates the new FBI for a specific region (here it's done by FWA but can be
#replaced with LGAs, etc)

#This code assumes the recalculated, modified data has NaNs where it has not been modified.
#Presumably, if it hasn't been modified and it's not NaNs, you're just replacing one value with an
#identical one...

#Edit Jan 24: To take multiple single model recalcs and apply all of them to the original data.

#This assumes the file "datacube_region_designate_rating.py"
#is in the same directory as this.

import sys, os
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

def replace_fuel_calc_rating(original_path, newdata_paths, date_in, area_mask, fuel_lut_pth_orig, fuel_lut_pth_new=None):
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
            #tidy up negative values (for some reason we use those instead of nans now...)
            official_max = official_max.where(official_max>=0)

            #Area mask to the desired region:
            clipped_orig = official_max.where(~area_mask.isnull())
            
            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_orig, 90)
            desig_rating = rating_calc(desig_fbi)
            
            #Find the fuel type that is most dominant in the 90th percentile.
            fuel_type_map_fbi = official_file_in['fuel_type']
            orig_dom_type = find_dominant_fuel_type_for_a_rating(clipped_orig, desig_fbi, fuel_type_map_fbi,fuel_lut_pth_orig)
            
            #Do the same to the recalculated data, iteratively
            clipped_replaced = xr.DataArray(coords=(clipped_orig['latitude'], clipped_orig['longitude']))
            for recalcs in newdata_paths:
                recalc_file_in = xr.open_dataset(recalcs+"VIC_"+date_str+"_recalc.nc")
                recalc_max = recalc_file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)
                recalc_max = recalc_max.where(recalc_max>=0)
                """
                recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) 
                recalc_max.rio.write_crs("EPSG:4326",inplace=True)
                clipped_recalc = recalc_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                """
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
    
    sys.path.append(os.path.abspath('./afdrs_fbi_recalc/scripts'))
    from helper_functions import loadGeoTiff, regrid_xr

    
    dc_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_mar25/recalc_files/'
    #recalc_path can be multiple paths for multiple changes
    recalc_path = ['C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/pine2025_updates/recalc_files/']

    path_to_fuel_lut_orig = "C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv"
    path_to_fuel_lut_recalc = "C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv"

    #Get the regional template grid for defining each area:
#    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_FWA_1500m.nc")
    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_LGA_1500m.nc")
#    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_ICC_1500m.nc")
    area_name_list = np.unique(map_by_pixel_in['Area_Name'])
    area_name_list = area_name_list[area_name_list!='']
    
    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
    
    #Get a list of the dates actually in the range by checking all the daily files are there.
    dates_used=[]
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(dc_path+'VIC_'+date_str+'_recalc.nc').is_file():
                dates_used.append(dt)
                if not Path(recalc_path[0]+'VIC_'+date_str+'_recalc.nc').is_file():
                    raise FileNotFoundError('Matching replacement day not found for '+date_str+'. Exiting')                   

    k=0
    for area_name in area_name_list:
        lgas_to_miss =['Darebin', 'Moreland', 'Maribyrnong', 'Yarra', 'Boroondara','Monash', 'Port Phillip', 
                       'Glen Eira', 'Bayside', 'Hobsons Bay', 'Moonee Valley',
                       'Stonnington', 
                       'Gabo Island',
                       'Kingston', 'Melbourne']
        if area_name in lgas_to_miss:
            print("Skip "+area_name+", is in metro or otherwise too small. Next.")
            continue
        print('Commencing '+area_name)    
        start_time = time.time()
        pool = mp.Pool(12)
        area_in = map_by_pixel_in['Area_Name'].where(map_by_pixel_in['Area_Name']==area_name)
        results_pool = [pool.apply_async(replace_fuel_calc_rating, args=(dc_path, recalc_path, dt, area_in, path_to_fuel_lut_orig, path_to_fuel_lut_recalc)) for dt in dates_used]    
        pool.close()
        pool.join()
        results_list_ = [r.get() for r in results_pool]
        end_time = time.time()
        print("Time taken: "+str(round(end_time-start_time, 3)))
        fbi_and_rating_per_area = pd.DataFrame(results_list_, columns=['Date', area_name+'_Original_FBI', area_name+'_Original_rating', area_name+'_Original_Dominant FT', area_name+'_Changed_FBI', area_name+'_Changed_rating', area_name+'_Changed_dominant FT'])
        if k==0:
            fbi_and_rating_changes = fbi_and_rating_per_area
        else:
            fbi_and_rating_changes = fbi_and_rating_changes.merge(fbi_and_rating_per_area, left_on='Date', right_on='Date', how='inner')
        k=k+1
        end_time = time.time()
        print('Time taken for this region: '+str(round(end_time-start_time, 3)))

    fbi_and_rating_changes.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_mar25/changes/fbi_pineupdates_lga.csv")
