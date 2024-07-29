#Replaces one or multiple fuel types in the historical (datacube) with recalculated FBI thresholds
#based on one of its variables.

#This is a faster method than "replace_fueltype_changes" since we don't need to recalculate
#the ratings using new fuel parameters, it's just using the existing outputs to calculate
#ratings based on new thresholds, using data already in the file.

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

def threshold_change_calc_rating(data_path, date_in, area_mask, fuel_lut_pth_orig, fuel_type_for_new_thresholds):
    from datacube_region_designate_rating import find_dominant_fuel_type_for_a_rating, rating_calc
    from fdrs_calcs.spread_models.fire_behaviour_index import _fbi_from_thresholds
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
            official_file_in = xr.open_dataset(data_path+'VIC_'+date_str+'_recalc.nc')
            
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
            
            #Do the same to a scenario with replaced FBIs. But first need to do the replacement
            #First: Get the fuel codes we want to replace, make the mask.
            lut_in = pd.read_csv(fuel_lut_pth_orig)
            fuel_codes_for_thresholds = lut_in[lut_in['FBM']==fuel_type_for_new_thresholds]['FTno_State']
            code_mask = xr.where(np.isin(fuel_type_map_fbi[3,:,:].where(~area_mask.isnull()), fuel_codes_for_thresholds), True, False)
            
            #Choose the variable we want to apply thresholds to - with mask applied:
            variable_to_calc_fbi = official_file_in['intensity'].where(code_mask).max(dim='time', skipna=True, keep_attrs=True).values
            thresholds = [0, 100, 750, 4000, 20000, 40000]
            var_max_ = 90000
            
            #Need to work out how to collapse to 1D before back to 2D using the mask.. no idea...
            #WAIT - it's all about using the mask to your advantage...
            fbi_new_np_arr = calc_fbi_from_thresholds(variable_to_calc_fbi, thresholds, var_max_)
            #convert to xarray
            fbi_new = xr.DataArray(fbi_new_np_arr, dims=['latitude', 'longitude'])
            
            clipped_replaced = fbi_new.fillna(clipped_orig)
            clipped_replaced.name = 'index_1'
            recalc_fbi = np.nanpercentile(clipped_replaced, 90)
            recalc_rating = rating_calc(recalc_fbi)
                        
            #Find most dominant fuel type for the recalculated data.
            dom_typ_recalc = find_dominant_fuel_type_for_a_rating(clipped_replaced, recalc_fbi, fuel_type_map_fbi, fuel_lut_pth_orig)
            
            official_file_in.close()
    except FileNotFoundError:
            print(date_str+" not found.")
            

    outputs_ = date_in, desig_fbi, desig_rating, orig_dom_type, recalc_fbi, recalc_rating, dom_typ_recalc
    
    return outputs_

def calc_fbi_from_thresholds(metric, thresholds, metric_max):
    # setup FBI array the same shape as metric
    FBI = np.full(metric.shape, np.nan)
    FBI_thresholds = [0, 6, 12, 24, 50, 100]
    FBI_HIGH = 200
    # use numpy.interp to do the interpolation with the thresholds
    mask = metric < thresholds[-1]
    FBI[mask] = np.interp(metric[mask], thresholds, FBI_thresholds)
    # above top threshold scale so that METRIC_HIGH has FBI of FBI_HIGH, but
    # continue rising above that if metric is greater than METRIC_HIGH
    mask = metric >= thresholds[-1]
    FBI[mask] = FBI_thresholds[-1] + (FBI_HIGH - FBI_thresholds[-1]) * (
        metric[mask] - thresholds[-1]
        ) / (metric_max - thresholds[-1])

    # round to nearest integer
    FBI = np.trunc(FBI)

    return FBI

    
if __name__=="__main__":
    dc_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/'
    #recalc_path can be multiple paths for multiple changes
    recalc_path = ['C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/allforest_canopyheight_changes/recalc_files/']

    path_to_fuel_lut_orig = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231012043244.csv"

    #Get the regional template grid for defining each area:
    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_FWA_1500m.nc")
    area_name_list = np.unique(map_by_pixel_in['Area_Name'])
    area_name_list = area_name_list[area_name_list!='']
    
    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
    
    #Set FBM to change thresholds:
    ft_for_threshold_change = 'Mallee heath'
    
    
    #Get a list of the dates actually in the range by checking all the daily files are there.
    dates_used=[]
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(dc_path+'VIC_'+date_str+'_recalc.nc').is_file():
                dates_used.append(dt)

    k=0
    for area_name in area_name_list:
        lgas_to_miss =['Darebin', 'Moreland', 'Maribyrnong', 'Yarra', 'Boroondara','Monash','Banyule', 'Port Phillip', 
                       'Glen Eira', 'Bayside', 'Hobsons Bay', 'Maroondah', 'Moonee Valley',
                       'Whitehorse','Stonnington', 'Falls Creek Alpine Resort', 'Lake Mountain Alpine Resort',
                       'Mount Buller Alpine Resort', 'Mount Hotham Alpine Resort', 'Mount Stirling Alpine Resort', 'Gabo Island',
                       'Kingston', 'Melbourne']
        if area_name in lgas_to_miss:
            print("Skip "+area_name+", is in metro or otherwise too small. Next.")
            continue
        print('Commencing '+area_name)
        area_in = map_by_pixel_in['Area_Name'].where(map_by_pixel_in['Area_Name']==area_name)
        start_time = time.time()
        """
        for dt in dates_used:
            print(dt)
            results_list_ = threshold_change_calc_rating(dc_path, dt, area_in, path_to_fuel_lut_orig, ft_for_threshold_change)
        """
        pool = mp.Pool(12)
        area_in = map_by_pixel_in['Area_Name'].where(map_by_pixel_in['Area_Name']==area_name)
        results_pool = [pool.apply_async(threshold_change_calc_rating, args=(dc_path, dt, area_in, path_to_fuel_lut_orig, ft_for_threshold_change)) for dt in dates_used]    
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

    fbi_and_rating_changes.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_jul24/changes/fwd/fbi_changes_malleethresh.csv")