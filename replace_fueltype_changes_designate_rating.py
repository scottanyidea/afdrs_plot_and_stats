#Replaces one or multiple fuel types in the historical (datacube) with a recalculated
#version that has been modified in some way, e.g. perturbed, updated or has bug fixes.
#Then calculates the new FBI for a specific region (here it's done by FWA but can be
#replaced with LGAs, etc)

#This code assumes the recalculated, modified data has NaNs where it has not been modified.
#Presumably, if it hasn't been modified and it's not NaNs, you're just replacing one value with an
#identical one...

import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
from datetime import datetime
import time

dc_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jan_24/recalc_files/'
recalc_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/mallee_only_25pccover_FL7_dec23_deltabug/recalc_files/'

#Set dates:
dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,4,30), freq='D')
#dates_ = pd.date_range(datetime(2017,10,4), datetime(2017,10,4), freq='D')

area_name = 'Mallee'

def replace_fuel_calc_rating(original_path, newdata_path, date_list, area_name):
    dates_used = []
    fbi_orig = []
    fbi_recalc = []
    rating_orig = []
    rating_recalc = []
    for dt in date_list:
        time_start = time.time()
        print('starting '+str(dt))
        try:
            #First get the files.
            date_str = dt.strftime("%Y%m%d")
            recalc_file_in = xr.open_dataset(newdata_path+"VIC_"+date_str+"_recalc.nc")
            official_file_in = xr.open_dataset(original_path+'VIC_'+date_str+'_recalc.nc')
        
            #Find maximum FBI along the day
            official_max = official_file_in['index_1'].max(dim='time',skipna=True, keep_attrs=True)
            recalc_max = recalc_file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)
        
            #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
            shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp", crs='ESPG:4326')
            area_polygon = shp_in[shp_in['Area_Name']==area_name]
            recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
            recalc_max.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
            official_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
            official_max.rio.write_crs("EPSG:4326",inplace=True)
            clipped_orig = official_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
            clipped_recalc = recalc_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)

            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_orig, 90)
            desig_rating = rating_calc(desig_fbi)
            fbi_orig.append(desig_fbi)
            rating_orig.append(desig_rating)

            #Replace the official data with the new data where it exists.
            clipped_recalc = clipped_recalc.fillna(clipped_orig)
            recalc_fbi = np.nanpercentile(clipped_recalc, 90)
            recalc_rating = rating_calc(recalc_fbi)
            fbi_recalc.append(recalc_fbi)
            rating_recalc.append(recalc_rating)
            
            #If we get to this point, grab the date.
            dates_used.append(dt)
            
            recalc_file_in.close()
            official_file_in.close()
        except FileNotFoundError:
            print(date_str+" not found. Skip to next")
            pass
        finally:
            time_end = time.time()
            print('Time for this iteration: '+str(time_end - time_start))
    rating_table = pd.DataFrame(list(zip(dates_used, fbi_orig, rating_orig, fbi_recalc, rating_recalc)), columns=['Date', 'Original FBI','Original rating', 'Changed FBI','Changed rating'])
    
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

fbi_and_rating = replace_fuel_calc_rating(dc_path, recalc_path, dates_, area_name)
fbi_and_rating.to_csv("C:/Users/clark/analysis1/Mallee_spreadsheet_calcs/csv_outputs/mallee_historical_fbi_rating_25pccoverFL7_grass35.csv")