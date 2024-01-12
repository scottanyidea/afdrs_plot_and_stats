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


def calc_region_rating(data_path, date_list, area_name):
    dates_used = []
    fbi_list = []
    rating_list = []
    fdi_list = []
    for dt in date_list:
        time_start = time.time()
        print('starting '+str(dt))
        try:
            #First get the files.
            date_str = dt.strftime("%Y%m%d")
            file_in = xr.open_dataset(data_path+"VIC_"+date_str+"_recalc.nc")
        
            #Find maximum FBI along the day
            recalc_max = file_in['index_1'].max(dim='time',skipna = True, keep_attrs=True)
        
            #Filter down to desired FWA. #EPSG:4326 corresponds to WGS 1984 CRS
            shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp", crs='ESPG:4326')
            area_polygon = shp_in[shp_in['Area_Name']==area_name]
            recalc_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
            recalc_max.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
            clipped_recalc = recalc_max.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)

            #Find the 90th percentile FBI for the original file.
            desig_fbi = np.nanpercentile(clipped_recalc, 90)
            desig_rating = rating_calc(desig_fbi)
            fbi_list.append(desig_fbi)
            rating_list.append(desig_rating)
            
            #If we get to this point, grab the date.
            dates_used.append(dt)
                        
            #Also get McArthur FDI.
            fdi_max = file_in['FDI_SFC'].max(dim='time',skipna=True,keep_attrs=True)
            fdi_max.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
            fdi_max.rio.write_crs("EPSG:4326",inplace=True)
            clipped_fdi = fdi_max.rio.clip(area_polygon.geometry.apply(mapping),area_polygon.crs,drop=False)
                
            fdi_list.append(np.nanpercentile(clipped_fdi, 90))
            file_in.close()

        except FileNotFoundError:
            print(date_str+" not found. Skip to next")
            pass
        finally:
            time_end = time.time()
            print('Time for this iteration: '+str(time_end - time_start))
    
    rating_table = pd.DataFrame(list(zip(dates_used, fbi_list, rating_list, fdi_list)), columns=['Date', 'FBI','Rating', 'McArthur_FDI'])

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

if __name__=="__main__":
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jan_24/recalc_files/'
    
    #Set dates:
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
        
    area_name = 'North East'

    fbi_and_rating = calc_region_rating(fbi_data_path, dates_, area_name)
    fbi_and_rating.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/"+area_name+"_datacube_20172022_fbi_rating.csv")
