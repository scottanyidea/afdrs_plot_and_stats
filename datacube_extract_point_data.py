#Extract statistics for a single lat, lon point in the datacube data.
#Could be adapted for VicClim etc

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from pathlib import Path


def extract_point_data(data_path, date_in, lat, lon):
    date_str = date_in.strftime("%Y%m%d")
    file_in = xr.open_dataset(data_path+"VIC_"+date_str+"_recalc.nc")

    #Find point closest to the coordinates given:
    point_sel = file_in.sel(longitude=lon, latitude=lat, method='nearest')
    fbi_max = point_sel['index_1'].max(dim='time').values
    rating = point_sel['rating_1'].max(dim='time').values
    
    result_list = date_in, fbi_max, rating
    
    return result_list

if __name__=="__main__":
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/'
    
    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')

    lat_in = -35.12
    lon_in = 142.00

    dates_used = []
    print('Getting dates')
    for dt in dates_:
       date_str = dt.strftime("%Y%m%d")
       if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
           dates_used.append(dt)
           
    results_list = []
    for dt in dates_used:
        print(dt)
        results_ = extract_point_data(fbi_data_path, dt, lat_in, lon_in)
        results_list.append(results_)
    
    fbi_and_rating = pd.DataFrame(results_list, columns=['Date', 'FBI', 'Rating'])
    fbi_and_rating.to_csv("point_dataframe_walpeup.csv")