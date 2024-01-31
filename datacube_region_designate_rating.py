#Calculates the new FBI and McArthur FDI for a specific region (here it's done by FWA but can be
#replaced with LGAs, etc)

import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
from datetime import datetime
import time

import multiprocessing as mp
from pathlib import Path

def calc_region_rating(data_path, date_in, area_shp, fuel_lut_path):
    """
    Designate an FBI, fire danger rating and McArthur FDI for a defined region and date.
    
    An FBI and rating is calculated based on the 90th percentile of pixels within that region,
    which is defined by the polygon input.
    
    The most dominant fuel type model that drives the FBI and rating is also returned as a
    string.
    
    This assumes the input file has a file name "VIC_<YYYYMMDD>_recalc.nc", and is hourly
    data for that date alone.

    Parameters
    ----------
    data_path (string) : Path to where the input data is located. 
    date_in (datetime object): Date to look at.
    area_shp (geopandas object) : Geopandas object with the polygon with which to calculate
        the designated FBI and rating within.
    fuel_lut_path (string) : Path and file name of the fuel lookup table to match the fuel type codes with 
    their model names.

    Returns
    -------
    result_list : List containing, in order:
        the date, 
        the FBI, 
        rating, 
        dominant fuel type model,
        McArthur FDI.

    """

    try:
            #First get the files.
            date_str = date_in.strftime("%Y%m%d")
            file_in = xr.open_dataset(data_path+"VIC_"+date_str+"_recalc.nc")
        
            #Find maximum FBI along the day
            recalc_max = file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)  
            
            #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
            recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
            recalc_max.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
            clipped_recalc = recalc_max.rio.clip(area_shp.geometry.apply(mapping), area_shp.crs, drop=False)

            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_recalc, 90)
            desig_rating = rating_calc(desig_fbi)
            
            #Find dominant model for each rating (ie. what's driven the 90th percentile)
            fuel_map = file_in['fuel_type']
            dom_typ_str = find_dominant_fuel_type_for_a_rating(clipped_recalc, desig_fbi, fuel_map, fuel_lut_path)
                        
            #Also get McArthur FDI.
            fdi_max = file_in['FDI_SFC'].max(dim='time',skipna=True,keep_attrs=True)
            fdi_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
            fdi_max.rio.write_crs("EPSG:4326",inplace=True)
            clipped_fdi = fdi_max.rio.clip(area_shp.geometry.apply(mapping),area_shp.crs,drop=False)
                
            desig_fdi = np.nanpercentile(clipped_fdi, 90)
            file_in.close()

    except FileNotFoundError:
            print(date_str+" not found. Exiting")
            
    result_list = date_in, desig_fbi, desig_rating, dom_typ_str, desig_fdi 
    return result_list
 
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
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jan_24/recalc_files/'
#    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/v2024.1b7/full_recalc_jan_24/recalc_files/'
     
    fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231012043244.csv"
    
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp"
    shp_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp"
    
    #Set dates:
#    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
    dates_ = pd.date_range(datetime(2021,10,17), datetime(2021,10,20), freq='D')        
    area_name = 'Knox'
    
    dates_used = []
    
    shp_in = geopandas.read_file(shp_path, crs='ESPG:4326')   
    
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
            dates_used.append(dt)
    
    #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
    shp_in = geopandas.read_file(shp_path, crs='ESPG:4326')
    area_polygon = shp_in[shp_in['Area_Name']==area_name]
    
    
    pool = mp.Pool(8)
    start_time = time.time()
    results_mp = [pool.apply_async(calc_region_rating, args=(fbi_data_path, date_s, area_polygon, fuel_lut_path)) for date_s in dates_used]
    pool.close()
    pool.join()
    """
    results_list = []
    for dt in dates_used:
        print(dt)
        results_ = calc_region_rating(fbi_data_path, dt, area_polygon, fuel_lut_path)
        results_list.append(results_)
    """
    end_time = time.time()
    print('Time taken for whole timeframe: '+str(end_time-start_time))

    #Data post processing for saving:
    results_list = [r.get() for r in results_mp]
    fbi_and_rating = pd.DataFrame(results_list, columns=['Date', 'FBI', 'Rating', 'Dominant FT', 'McArthur FDI'])
    
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_feb24/"+area_name+"_datacube_2017-2022_fbi_rating.csv")
