#Calculates the new FBI and McArthur FDI for a specific region (here it's done by FWA but can be
#replaced with LGAs, etc)

#This one is to do all areas in a shapefile, e.g. LGA, FWD. This assumes the file "datacube_region_designate_rating.py"
#is in the same directory as this.

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

from datacube_region_designate_rating import calc_region_rating
import sys, os
sys.path.append(os.path.abspath('./afdrs_fbi_recalc-main/scripts'))
from helper_functions import loadGeoTiff, regrid_xr

if __name__=="__main__":
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/'
#    fbi_data_path = 'M:/Archived/AFDRS/VIC_Grids/'
#    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/v2024.1b7/full_recalc_jan_24/recalc_files/'
     
    fuel_lut_path = "C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv"
    
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp"
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90309_VIC_Boundary_SHP_ICC\PID90309_VIC_Boundary_SHP_ICC.shp"
#     shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA\PID90409_VIC_Boundary_SHP_LGA.shp"
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90209_VIC_Boundary_SHP_CFA\PID90209_VIC_Boundary_SHP_CFA.shp"
#    shp_path = "C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90509_VIC_Boundary_SHP_PDD\PID90509_VIC_Boundary_SHP_PDD.shp"

    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
#    dates_ = pd.date_range(datetime(2017,10,4), datetime(2017,10,5), freq='D')        

    #Which dates actually exist.
    dates_used = []
    print('Getting dates')
    for dt in dates_:
        date_str = dt.strftime("%Y%m%d")
        if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
            dates_used.append(dt)
    
    print('Dates retrieved.')
#    shp_in = geopandas.read_file(shp_path, crs='ESPG:4326')  
    k=0
    
    #Get template for mapping regions:
    map_by_pixel_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_ICC_1500m.nc")
    area_list = np.unique(map_by_pixel_in['Area_Name'].values)  #get unique area names
    area_list = np.delete(area_list, np.where(area_list==''))  #remove the empty area
    
    #Get the McArthur mask:
    mcarthur_mask = loadGeoTiff("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/v4_forest0grass1malleeheath1heathland0.tif",da_var_name='grass_forest_mask', as_Dataset = True)
    #Grab a vicgrids file to line up the grids:
    file_in = xr.open_dataset(fbi_data_path+"VIC_20180101_recalc.nc")
#    file_in = xr.open_dataset(fbi_data_path+"VIC_20180101.nc")
    mcarthur_mask = regrid_xr(file_in, mcarthur_mask, method = "nearest")
    file_in.close()
    
#    for area_name in shp_in['Area_Name']:
    for area_name in area_list:
        lgas_to_miss =['Darebin', 'Moreland', 'Maribyrnong', 'Yarra', 'Boroondara','Monash','Port Phillip', 
                       'Glen Eira', 'Bayside', 'Hobsons Bay', 'Moonee Valley',
                       'Whitehorse','Stonnington',
                       'Melbourne', 'Kingston']
        if area_name in lgas_to_miss:
            print("Skip "+area_name+", is in metro or otherwise too small. Next.")
            continue
        print('Commencing '+area_name)    
        start_time = time.time()
        map_by_pixel = map_by_pixel_in['Area_Name'].where(map_by_pixel_in['Area_Name']==area_name)
        pool = mp.Pool(12)
        #area_polygon = shp_in[shp_in['Area_Name']==area_name]
        mp_outputs_ = [pool.apply_async(calc_region_rating, args=(fbi_data_path, dt, map_by_pixel, fuel_lut_path, mcarthur_mask)) for dt in dates_used]
        pool.close()
        pool.join()
        results_list_fbi = [r.get() for r in mp_outputs_]
        fbi_rating_per_area = pd.DataFrame(results_list_fbi, columns=['Date', area_name+'_FBI',area_name+'_Rating', area_name+'_Dominant FT', area_name+'_McArthur_FDI', area_name+'_Forest_or_grass_FDI'])
        if k==0:
            fbi_and_rating = fbi_rating_per_area
        else:
            fbi_and_rating = fbi_and_rating.merge(fbi_rating_per_area, left_on='Date', right_on='Date', how='inner')
        k=k+1
        end_time = time.time()
        print('Time taken for this region: '+str(round(end_time-start_time, 3)))
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_jul24/tables/datacube_2017-2022_fbi_rating_icc.csv")
