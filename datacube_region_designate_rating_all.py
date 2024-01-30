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

from datacube_region_designate_rating import rating_calc

import multiprocessing as mp
from pathlib import Path

def calc_region_rating(data_path, date_in, area_polygon, fuel_lut_path):
    from replace_fueltype_changes_designate_rating import find_dominant_fuel_type_for_a_rating

    try:
            #First get the files.
            date_str = date_in.strftime("%Y%m%d")
            file_in = xr.open_dataset(data_path+"VIC_"+date_str+"_recalc.nc")
        
            #Find maximum FBI along the day
            recalc_max = file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)
        
            #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS

            recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
            recalc_max.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
            clipped_recalc = recalc_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)

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
            clipped_fdi = fdi_max.rio.clip(area_polygon.geometry.apply(mapping),area_polygon.crs,drop=False)
                
            fdi_out = np.nanpercentile(clipped_fdi, 90)
            file_in.close()

    except FileNotFoundError:
            print(date_str+" not found. Exiting")
    result_list = date_in, desig_fbi, desig_rating, dom_typ_str, fdi_out
    return  result_list

if __name__=="__main__":
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jan_24/recalc_files/'
#    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/v2024.1b7/full_recalc_jan_24/recalc_files/'
     
    fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231012043244.csv"
    
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp"
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90309_VIC_Boundary_SHP_ICC\PID90309_VIC_Boundary_SHP_ICC.shp"
    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp"

    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
#    dates_ = pd.date_range(datetime(2017,10,4), datetime(2017,10,5), freq='D')        

    #Which dates actually exist.
    dates_used = []
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
            dates_used.append(dt)

    shp_in = geopandas.read_file(shp_path, crs='ESPG:4326')  
    k=0
    
    for area_name in shp_in['Area_Name']:
        lgas_to_miss =['Darebin', 'Moreland', 'Maribyrnong', 'Yarra', 'Boroondara','Monash','Banyule', 'Port Phillip', 
                       'Glen Eira', 'Bayside', 'Hobsons Bay', 'Maroondah', 'Moonee Valley',
                       'Whitehorse','Stonnington', 'Falls Creek Alpine Resort', 'Lake Mountain Alpine Resort',
                       'Mount Buller Alpine Resort', 'Mount Hotham Alpine Resort', 'Mount Stirling Alpine Resort', 'Gabo Island',
                       'Greater Dandenong', 'Kingston', 'Knox','Erica']
        if area_name in lgas_to_miss:
            print("Skip "+area_name+", is in metro or otherwise too small. Next.")
            continue
        print('Commencing '+area_name)    
        start_time = time.time()
        pool = mp.Pool(12)
        area_polygon = shp_in[shp_in['Area_Name']==area_name]
        mp_outputs_ = [pool.apply_async(calc_region_rating, args=(fbi_data_path, dt, area_polygon, fuel_lut_path)) for dt in dates_used]
        pool.close()
        pool.join()
        results_list_fbi = [r.get() for r in mp_outputs_]
        fbi_rating_per_area = pd.DataFrame(results_list_fbi, columns=['Date', area_name+'_FBI',area_name+'_Rating', area_name+'_Dominant FT', area_name+'_McArthur_FDI'])
        if k==0:
            fbi_and_rating = fbi_rating_per_area
        else:
            fbi_and_rating = fbi_and_rating.merge(fbi_rating_per_area, left_on='Date', right_on='Date', how='inner')
        k=k+1
        end_time = time.time()
        print('Time taken for this region: '+str(round(end_time-start_time, 3)))
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_feb24/datacube_2017-2022_fbi_rating_lga.csv")
