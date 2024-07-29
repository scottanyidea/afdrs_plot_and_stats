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
import sys, os

import multiprocessing as mp
from pathlib import Path

sys.path.append(os.path.abspath('./afdrs_fbi_recalc-main/scripts'))
from helper_functions import loadGeoTiff, regrid_xr

def calc_region_rating(data_path, date_in, area_mask, fuel_lut_path, mcarthur_mask, calc_fdi=True):
#def calc_region_rating(data_path, date_in, area_shp, fuel_lut_path, mcarthur_mask):
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
#            file_in = xr.open_dataset(data_path+"VIC_"+date_str+".nc")
        
            #Find maximum FBI along the day
            recalc_max = file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)  
            recalc_max = recalc_max.where(recalc_max>=0)
            #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
            
            clipped_recalc = recalc_max.where(~area_mask.isnull())
            """
            recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
            recalc_max.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
            clipped_recalc = recalc_max.rio.clip(area_shp.geometry.apply(mapping), area_shp.crs, drop=False)
            """
            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_recalc, 90)
            desig_rating = rating_calc(desig_fbi)
            
            #Find dominant model for each rating (ie. what's driven the 90th percentile)
            fuel_map = file_in['fuel_type']
            dom_typ_str = find_dominant_fuel_type_for_a_rating(clipped_recalc, desig_fbi, fuel_map, fuel_lut_path)
                        
            #Also get McArthur FDI.
            if calc_fdi==True:
                fdi_max = file_in['FDI_SFC'].max(dim='time',skipna=True,keep_attrs=True)
            
                clipped_fdi = fdi_max.where(~area_mask.isnull())
                """
                fdi_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
                fdi_max.rio.write_crs("EPSG:4326",inplace=True)
                clipped_fdi = fdi_max.rio.clip(area_shp.geometry.apply(mapping),area_shp.crs,drop=False)
                """
                desig_fdi = np.nanpercentile(clipped_fdi, 90)

                #And also get dominant FFDI or GFDI.
                dom_mcarthur_str = find_dominant_ffdi_or_gfdi(clipped_fdi, desig_fdi, mcarthur_mask)
                
                result_list = date_in, desig_fbi, desig_rating, dom_typ_str, desig_fdi, dom_mcarthur_str
            else:
                result_list = date_in, desig_fbi, desig_rating, dom_typ_str
            file_in.close()

    except FileNotFoundError:
            print(date_str+" not found. Exiting")
            
    
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
    #This assumes the fbi_array is already clipped to the desired geometry and is 2D (ie. already maximum)
    
    #Mask fuel type map to be same as the FBI map:
    fuel_type_map_clipped = xr.where(fbi_arr, fuel_type_map[0,:,:], np.nan)
    fuel_type_map_clipped.name = 'fuel_type'
    
    #Merge FBI with fuel types, and mask to only those pixels above 90th percentile (or whatever we set rating_val to)
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
        topmodel = 'None dominant'
    else:
        topmodel = top_pixels_table.FBM.value_counts().index[0]
        
    return topmodel  #ie. return the NAME of the top fuel type

def find_dominant_ffdi_or_gfdi(fdi_arr, fdi_rating_val, mcarthur_mask):
    #assume FBI and FFDI are already clipped to the desired geometry. 
    #Based on logic of this function - not necessary to do so for FFDI array.
    #assume ffdi_arr is a 2D xarray (ie. maximum)
    
    #Mask FDI to only those pixels above 90th percentile (or whatever we set fdi_rating_val to)
    fdi_arr_masked = fdi_arr.where((fdi_arr>=fdi_rating_val))

    #Mask FFDI to the same as FDI
    mcarthur_mask_trim = xr.where(~np.isnan(fdi_arr_masked), mcarthur_mask['grass_forest_mask'], np.nan)
    ffdi_masked = xr.where(mcarthur_mask_trim==0, fdi_arr_masked, np.nan)
    
    #Some pixels in FFDI will be nans because they are instead grass. We use this to count.
    ffdi_pixel_frac = ffdi_masked.count()/fdi_arr_masked.count()
    
    #If FFDI frac is >0.5, forest is dominant. Else it's grass.
    if ffdi_pixel_frac>0.5:
        dom_index = 'Forest'
    else:
        dom_index = 'Grass'
        
    return dom_index

if __name__=="__main__":
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/'
#    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/v2024.1b7/full_recalc_jan_24/recalc_files/'
#    fbi_data_path = 'M:/Archived/AFDRS/VIC_Grids/'
    fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv"
    
    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp"
#    shp_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp"
    
    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
#    dates_ = pd.date_range(datetime(2019,1,1), datetime(2019,1,5), freq='D')        

    #Get the regional template grid for defining each area:
    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_FWA_1500m.nc")
    area_name = 'East Gippsland'
    map_by_pixel = map_by_pixel_in['Area_Name'].where(map_by_pixel_in['Area_Name']==area_name)
    
    dates_used = []
    
    shp_in = geopandas.read_file(shp_path, crs='ESPG:4326')   
    
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
#        if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
            dates_used.append(dt)
    
    #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
    shp_in = geopandas.read_file(shp_path, crs='ESPG:4326')
    area_polygon = shp_in[shp_in['Area_Name']==area_name]
    
    #Get the McArthur mask:
    mcarthur_mask = loadGeoTiff("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/v4_forest0grass1malleeheath1heathland0.tif",da_var_name='grass_forest_mask', as_Dataset = True)
    #Grab a vicgrids file to line up the grids:
    file_in = xr.open_dataset(fbi_data_path+"VIC_20180101_recalc.nc")
#    file_in = xr.open_dataset(fbi_data_path+"VIC_20180101.nc")
    mcarthur_mask = regrid_xr(file_in, mcarthur_mask, method = "nearest")
    file_in.close()
    
        
    pool = mp.Pool(12)
    start_time = time.time()
    results_mp = [pool.apply_async(calc_region_rating, args=(fbi_data_path, date_s, map_by_pixel, fuel_lut_path, mcarthur_mask)) for date_s in dates_used]
    pool.close()
    pool.join()
    """
    start_time=time.time()
    results_list = []
    for dt in dates_used:
        print(dt)
        results_ = calc_region_rating(fbi_data_path, dt, map_by_pixel, fuel_lut_path, mcarthur_mask, calc_fdi=False)
        results_list.append(results_)
    """
    end_time = time.time()
    print('Time taken for whole timeframe: '+str(end_time-start_time))

    #Data post processing for saving:
    results_list = [r.get() for r in results_mp]
    fbi_and_rating = pd.DataFrame(results_list, columns=['Date', 'FBI', 'Rating', 'Dominant FT', 'McArthur_FDI', 'Forest_or_grass_FDI'])
    
#    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_mar24_2/data_tables/"+area_name+"_datacube_2017-2022_fbi_ratingold.csv")
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_jul24/"+area_name+"_datacube_2017-2022_fbi.csv")