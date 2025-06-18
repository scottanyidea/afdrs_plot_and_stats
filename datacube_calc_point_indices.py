#For a single lat, lon point
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import fdrs_calcs

def calc_point_fbi(data_path, date_in, lat, lon, fuel_lut, fuel_type=None):
    date_str = date_in.strftime("%Y%m%d")
    file_in = xr.open_dataset(data_path+"VIC_"+date_str+"_recalc.nc")

    #Find point closest to the coordinates given:
    point_sel = file_in.sel(longitude=lon, latitude=lat, method='nearest')

    if fuel_type is None:
        fuel_type_in = point_sel['fuel_type'].values
    else:
        fuel_type_in = np.full(len(point_sel['T_SFC'].values), fuel_type)
    
    calculated_fdrs_output_np_arr = fdrs_calcs.calculate_indicies(
        temp = point_sel['T_SFC'].values.reshape(-1),
        kbdi = point_sel['KBDI_SFC'].values.reshape(-1),
        sdi = point_sel['SDI_SFC'].values.reshape(-1),
        windmag10m = point_sel['WindMagKmh_SFC'].values.reshape(-1),
        rh = point_sel['RH_SFC'].values.reshape(-1),
        td = point_sel['Td_SFC'].values.reshape(-1),
        df = point_sel['DF_SFC'].values.reshape(-1),
        curing = point_sel['Curing_SFC'].values.reshape(-1),
        grass_fuel_load = point_sel['GrassFuelLoad_SFC'].values.reshape(-1),
        grass_condition= point_sel['grass_condition'].values.reshape(-1),
        precip = point_sel['precipitation'].values.reshape(-1),
        time_since_rain = point_sel['time_since_rain'].values.reshape(-1),
        time_since_fire = point_sel['time_since_fire'].values.reshape(-1),
        ground_moisture = np.full(len(point_sel['T_SFC']), np.nan),
        fuel_type = fuel_type_in,
        fuel_table = fuel_lut,
        hours = point_sel['hours'].values.reshape(-1),
        months = point_sel['months'].values.reshape(-1),
        )
    
    ffdi_arr = 2 * np.exp(-0.45 + 0.987 * np.log(point_sel['DF_SFC']) - 0.0345 * point_sel['RH_SFC'] + 0.0338 * point_sel['T_SFC'] + 0.0234 * point_sel['WindMagKmh_SFC'])

    gfdi_arr = np.exp(
        -1.523
        + 1.027 * np.log(point_sel['GrassFuelLoad_SFC'])
        - 0.009432 * np.power((100 - point_sel['Curing_SFC']), 1.536)
        + 0.02764 * point_sel['T_SFC'] 
        + 0.6422 * np.power(point_sel['WindMagKmh_SFC'], 0.5) 
        - 0.2205 * np.power(point_sel['RH_SFC'], 0.5)
    )
    result_list = [file_in['time'].values[0:23], calculated_fdrs_output_np_arr['index_1'][0:23], calculated_fdrs_output_np_arr['rating_1'][0:23], ffdi_arr.values[0:23], gfdi_arr.values[0:23]]
    
    return result_list

if __name__=="__main__":
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_mar25/recalc_files/'
    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv")

    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')

    lat_in = -35.1201
    lon_in = 142.0040

    dates_used = []
    print('Getting dates')
    for dt in dates_:
       date_str = dt.strftime("%Y%m%d")
       if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
           dates_used.append(dt)
           
    
    #Unfortunately trying to deal with outputs is a bit messy. so - the numpy arrays get 
    #appended to the list...
    times_list = []
    fbi_list = []
    rating_list = []
    ffdi_list = []
    gfdi_list = []
    
    for dt in dates_used:
        print(dt)
        results_ = calc_point_fbi(fbi_data_path, dt, lat_in, lon_in, fuel_lut, fuel_type=3061)
        
        times_list.append(dt.date())
        fbi_list.append(np.max(results_[1]))
        rating_list.append(np.max(results_[2]))
        ffdi_list.append(np.max(results_[3]))
        gfdi_list.append(np.max(results_[4]))
    """
    #...now we have a list of arrays for each day. Concatenate to create a single (long) record.
    times_list = np.concatenate(times_list, axis=0)+np.timedelta64(11, 'h') #add 11 hours to get to local (summer!) time
    fbi_list = np.concatenate(fbi_list, axis=0)
    rating_list = np.concatenate(rating_list, axis=0)
    ffdi_list = np.concatenate(ffdi_list, axis=0)
    gfdi_list = np.concatenate(gfdi_list, axis=0)
    """        
    fbi_and_rating_max = pd.DataFrame(data={'Date': times_list, 'FBI': fbi_list, 'rating': rating_list, 'FFDI':ffdi_list, 'GFDI': gfdi_list})
    fbi_and_rating_max.Date = pd.to_datetime(fbi_and_rating_max.Date)
    fbi_and_rating_max.to_csv('C:/Users/clark/analysis1/datacube_daily_stats/version_mar25/output_data/point_fbis_walpeup_dailymax.csv')

#%%
    #Section to merge the data - don't always need to run this
    dc_deecadistrict90 = pd.read_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_mar25/tables/datacube_2017-2022_fbi_rating_deecadistrict_90pct.csv", parse_dates=['Date'])
    dc_deecadistrict50 = pd.read_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_mar25/tables/datacube_2017-2022_fbi_rating_deecadistrict_50pct.csv", parse_dates=['Date'])
    """    
    dc_choice90 = dc_deecadistrict90[['Date', 'MURRAY GOLDFIELDS_FBI', 'MURRAY GOLDFIELDS_McArthur_FFDI']].rename(columns={'MURRAY GOLDFIELDS_FBI': 'FBI_90pct', 'MURRAY GOLDFIELDS_McArthur_FFDI': 'FDI_90pct'})
    dc_choice50 = dc_deecadistrict50[['Date', 'MURRAY GOLDFIELDS_FBI', 'MURRAY GOLDFIELDS_McArthur_FFDI']].rename(columns={'MURRAY GOLDFIELDS_FBI': 'FBI_50pct', 'MURRAY GOLDFIELDS_McArthur_FFDI': 'FDI_50pct'})
    """
    dc_choice90 = dc_deecadistrict90[['Date', 'MALLEE_FBI', 'MALLEE_McArthur_FFDI']].rename(columns={'MALLEE_FBI': 'FBI_90pct', 'MALLEE_McArthur_FFDI': 'FDI_90pct'})
    dc_choice50 = dc_deecadistrict50[['Date', 'MALLEE_FBI', 'MALLEE_McArthur_FFDI']].rename(columns={'MALLEE_FBI': 'FBI_50pct', 'MALLEE_McArthur_FFDI': 'FDI_50pct'})
    
    fbi_and_rating_max = fbi_and_rating_max.rename(columns={'FBI': 'FBI_point_Walpeup_3061', 'FFDI': 'FFDI_point_Walpeup'})
    
    df_final = pd.merge(left=dc_choice90, right=dc_choice50, left_on='Date', right_on='Date', how='inner')
    df_final = pd.merge(left=df_final, right=fbi_and_rating_max[['Date', 'FBI_point_Walpeup_3061', 'FFDI_point_Walpeup']], left_on='Date', right_on='Date', how='inner')
    df_final.to_csv('C:/Users/clark/analysis1/datacube_daily_stats/version_mar25/output_data/daily_FBI_FFDI_Mallee.csv')
