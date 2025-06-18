#For all lat/lon points corresponding to all AWS locations in Vic.

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import fdrs_calcs
from compile_archived_observations import make_observation_table_from_archive
import multiprocessing as mp

def calc_point_fbi(file_in_path, date_in, lat, lon, fuel_lut, fuel_type=None, calc_mcarthur=True):
    
    date_str = date_in.strftime("%Y%m%d")
    file_in_daily = xr.open_dataset(file_in_path+"VIC_"+date_str+"_recalc.nc")
    #Find point closest to the coordinates given:
    point_sel = file_in_daily.sel(longitude=lon, latitude=lat, method='nearest')
    
    if fuel_type is None:
        fuel_type_in = point_sel['fuel_type'].values
    else:
        fuel_type_in = np.full(point_sel['T_SFC'].shape, fuel_type)
    """
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
    """
    calculated_fdrs_output_np_arr = fdrs_calcs.calculate_indicies(
        temp = point_sel['T_SFC'].values,
        kbdi = point_sel['KBDI_SFC'].values,
        sdi = point_sel['SDI_SFC'].values,
        windmag10m = point_sel['WindMagKmh_SFC'].values,
        rh = point_sel['RH_SFC'].values,
        td = point_sel['Td_SFC'].values,
        df = point_sel['DF_SFC'].values,
        curing = point_sel['Curing_SFC'].values,
        grass_fuel_load = point_sel['GrassFuelLoad_SFC'].values,
        #grass_condition= point_sel['grass_condition'].values,
        grass_condition= np.full_like(point_sel['T_SFC'].values, 2),
        precip = point_sel['precipitation'].values,
        time_since_rain = point_sel['time_since_rain'].values,
        time_since_fire = point_sel['time_since_fire'].values,
        ground_moisture = np.full_like(point_sel['T_SFC'].values, np.nan),
        fuel_type = fuel_type_in,
        fuel_table = fuel_lut,
        hours = point_sel['hours'].values,
        months = point_sel['months'].values,
        )
    if calc_mcarthur==True:
        ffdi_arr = 2 * np.exp(-0.45 + 0.987 * np.log(point_sel['DF_SFC']) - 0.0345 * point_sel['RH_SFC'] + 0.0338 * point_sel['T_SFC'] + 0.0234 * point_sel['WindMagKmh_SFC'])

        gfdi_arr = np.exp(
            -1.523
            + 1.027 * np.log(point_sel['GrassFuelLoad_SFC'])
            - 0.009432 * np.power((100 - point_sel['Curing_SFC']), 1.536)
            + 0.02764 * point_sel['T_SFC'] 
            + 0.6422 * np.power(point_sel['WindMagKmh_SFC'], 0.5) 
            - 0.2205 * np.power(point_sel['RH_SFC'], 0.5)
            )
        result_list = [file_in_daily['time'].values[0:23], calculated_fdrs_output_np_arr['index_1'][0:23], calculated_fdrs_output_np_arr['rating_1'][0:23], ffdi_arr.values[0:23], gfdi_arr.values[0:23]]
    else:
        result_list = [file_in_daily['time'].values[0:23], calculated_fdrs_output_np_arr['index_1'][0:23], calculated_fdrs_output_np_arr['rating_1'][0:23]]

    return result_list

if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print("In this script, I've ignored a deprecation warning. Please fix at some point")
    
    #Set up paths, dates:
    fbi_data_path = 'C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_mar25/recalc_files/'
    fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv")

    dates_ = pd.date_range(datetime(2017,10,1), datetime(2022,5,31), freq='D')
    
    #Get all the AWS locations in Vic.
    #These dates are literally just to make the function work... all we want are the lat and lons
    start_date = datetime(year=2024,month=2,day=3,hour=6,minute=0,second=0)
    end_date = datetime(year=2024, month=2,day=3,hour=6,minute=55,second=59)
    
    obs_list_short = make_observation_table_from_archive(start_date, end_date)
    station_locs = obs_list_short[['station_full', 'latitude','longitude']].drop_duplicates(subset='station_full')
    
    #OK this gives us all the stations in the country... I think this gets most of Vic?
    stations_to_pick = ["AIREYS INLET", "ALBURY AIRPORT AWS", "AVALON AIRPORT", "BAIRNSDALE AIRPORT",
                        "BALLARAT AERODROME", "BEN NEVIS", "BENDIGO AIRPORT", "CAPE NELSON LIGHTHOUSE", 
                        "CAPE OTWAY LIGHTHOUSE", "CASTERTON", "CERBERUS", "CHARLTON", 
                        "COLDSTREAM", "COMBIENBAR AWS", "DARTMOOR", "EAST SALE AIRPORT", 
                        "EILDON FIRE TOWER", "EDENHOPE AIRPORT", "ESSENDON AIRPORT", 
                        "FALLS CREEK", "FERNY CREEK", "FRANKSTON (BALLAM PARK)", "GELANTIPY", 
                        "GEELONG RACECOURSE", "HUNTERS HILL", "HAMILTON AIRPORT","HOPETOUN AIRPORT", 
                        "HORSHAM AERODROME","KILMORE GAP", "KANAGULK", 
                        "KYABRAM", "LATROBE VALLEY AIRPORT", "LAVERTON RAAF", "LONGERENONG", 
                        "MALLACOOTA","MANGALORE AIRPORT", "MELBOURNE AIRPORT", "MILDURA AIRPORT", 
                        "MOORABBIN AIRPORT",
                        "MOUNT BULLER", "MOUNT BAW BAW", "MOUNT GELLIBRAND", "MOUNT HOTHAM", "MOUNT HOTHAM AIRPORT", 
                        "MOUNT WILLIAM", 
                        "MOUNT NOWA NOWA", "MORTLAKE RACECOURSE", "NILMA NORTH (WARRAGUL)", 
                        "NHILL AERODROME", 
                        "MOUNT MOORNAPA", "OMEO", "ORBOST", "PORT FAIRY AWS", "PORTLAND NTC AWS", 
                        "PUCKAPUNYAL WEST (DEFENCE)",
                        "REDESDALE", "RHYLL", "RUTHERGLEN RESEARCH",
                        "SCORESBY RESEARCH INSTITUTE", "SHEOAKS", "SHEPPARTON AIRPORT", "STAWELL AERODROME",
                        "SWAN HILL AERODROME", "TATURA INST SUSTAINABLE AG", "VIEWBANK", "WALPEUP RESEARCH", 
                        "WANGARATTA AERO", "WARRACKNABEAL AIRPORT", "WARRNAMBOOL AIRPORT NDB", "WESTMERE", 
                        "WILSONS PROMONTORY LIGHTHOUSE", "YANAKIE", "YARRAWONGA"]
    
    station_locs = station_locs[station_locs['station_full'].isin(stations_to_pick)]
    #List of lats and lons to use:
#    lat_in = station_locs['latitude'].values
#    lon_in = station_locs['longitude'].values

    #Get actual dates in VicGrid data since there are gaps
    dates_used = []
    k=0
    print('Getting dates')
    for dt in dates_:
       date_str = dt.strftime("%Y%m%d")
       if Path(fbi_data_path+'VIC_'+date_str+'_recalc.nc').is_file():
           dates_used.append(dt)
    
    #Calculate for each date and location.
    station_list = []
    times_list = []
    fbi_list = []
    fbi2_list = []
    fbimal_list = []
    ffdi_list = []
    gfdi_list = []

    print("For Mallee Heath - do only a subset of stations")
    stations_mallee = ["WALPEUP RESEARCH", "HOPETOUN AIRPORT", "NHILL AERODROME"]
    times_list3 = []
    station_list3 = []
    for stn in stations_mallee:
        print("Starting "+stn)
        lat_in = station_locs[station_locs['station_full']==stn]['latitude'].values
        lon_in = station_locs[station_locs['station_full']==stn]['longitude'].values
        pool = mp.Pool(10)
        pool_outputs = [pool.apply_async(calc_point_fbi, args=(fbi_data_path, dt, lat_in, lon_in, fuel_lut), kwds={'fuel_type':3049, 'calc_mcarthur':False}, ) for dt in dates_used]
        pool.close()
        pool.join()
        outputs_list = [r.get() for r in pool_outputs]
        #How this works: For each day, get the datetime list out of outputs_list, grab the last timestamp (because it's UTC),
        #then convert it to a date.
        times_list3.append([pd.to_datetime(outputs_list[i][0][-1]).date() for i in range(0, len(outputs_list))])   
        station_list3.append(np.full(len(outputs_list), stn))
        fbimal_list.append([np.max(outputs_list[i][1]) for i in range(0, len(outputs_list))])
    
    print("*********************************")
    print("Now starting Grass.")
    for stn in stations_to_pick:
        print("Starting "+stn)
        lat_in = station_locs[station_locs['station_full']==stn]['latitude'].values
        lon_in = station_locs[station_locs['station_full']==stn]['longitude'].values
        pool = mp.Pool(10)
        pool_outputs = [pool.apply_async(calc_point_fbi, args=(fbi_data_path, dt, lat_in, lon_in, fuel_lut), kwds={'fuel_type':3046} ) for dt in dates_used]
        pool.close()
        pool.join()
        outputs_list = [r.get() for r in pool_outputs]
        #How this works: For each day, get the datetime list out of outputs_list, grab the last timestamp (because it's UTC),
        #then convert it to a date.
        times_list.append([pd.to_datetime(outputs_list[i][0][-1]).date() for i in range(0, len(outputs_list))])   
        station_list.append(np.full(len(outputs_list), stn))
        fbi_list.append([np.max(outputs_list[i][1]) for i in range(0, len(outputs_list))])
        ffdi_list.append([np.max(outputs_list[i][3]) for i in range(0, len(outputs_list))])
        gfdi_list.append([np.max(outputs_list[i][4]) for i in range(0, len(outputs_list))])

    print("Now for the forest run.")
    times_list2 = []
    station_list2 = []
    for stn in stations_to_pick:
        print("Starting "+stn)
        lat_in = station_locs[station_locs['station_full']==stn]['latitude'].values
        lon_in = station_locs[station_locs['station_full']==stn]['longitude'].values
        pool = mp.Pool(10)
        pool_outputs = [pool.apply_async(calc_point_fbi, args=(fbi_data_path, dt, lat_in, lon_in, fuel_lut), kwds={'fuel_type':3007, 'calc_mcarthur':False}, ) for dt in dates_used]
        pool.close()
        pool.join()
        outputs_list = [r.get() for r in pool_outputs]
        #How this works: For each day, get the datetime list out of outputs_list, grab the last timestamp (because it's UTC),
        #then convert it to a date.
        times_list2.append([pd.to_datetime(outputs_list[i][0][-1]).date() for i in range(0, len(outputs_list))])   
        station_list2.append(np.full(len(outputs_list), stn))
        fbi2_list.append([np.max(outputs_list[i][1]) for i in range(0, len(outputs_list))])
    


    #I ended up with a list of lists in the above loops. Convert to flat lists then put into a pandas df:
    times_list = [x for xs in times_list for x in xs]
    station_list = [x for xs in station_list for x in xs]
    fbi_list = [x for xs in fbi_list for x in xs]
    fbi2_list = [x for xs in fbi2_list for x in xs]
    ffdi_list = [x for xs in ffdi_list for x in xs]
    gfdi_list = [x for xs in gfdi_list for x in xs]
    
    #Create DF:
    fbi_and_rating_max = pd.DataFrame(data={'Date': times_list, 'Station':station_list, 'Grass_FBI': fbi_list, 'FFDI':ffdi_list, 'GFDI': gfdi_list})
    fbi_and_rating_max.Date = pd.to_datetime(fbi_and_rating_max.Date)
    
    #Create DF of the forest FBI and merge on
    #This is because there is no guarantee the dates will remain in order going from one loop to the next...
    times_list2 = [x for xs in times_list2 for x in xs]
    station_list2 = [x for xs in station_list2 for x in xs]

    fbi_and_rating_max_forest = pd.DataFrame(data={'Date': times_list2, 'Station':station_list2, 'Forest_FBI': fbi2_list})
    fbi_and_rating_max_forest.Date = pd.to_datetime(fbi_and_rating_max_forest.Date)
    
    fbi_and_rating_max = pd.merge(left=fbi_and_rating_max, right=fbi_and_rating_max_forest, left_on=['Date', 'Station'], right_on=['Date', 'Station'], how='inner')
    
    #Finally get the Mallee Heath merged.
    times_list3 = [x for xs in times_list3 for x in xs]
    station_list3 = [x for xs in station_list3 for x in xs]
    fbimal_list =[x for xs in fbimal_list for x in xs]
    fbi_max_mh = pd.DataFrame(data={'Date': times_list3, 'Station':station_list3, 'MalleeHeath_FBI': fbimal_list})
    fbi_max_mh.Date = pd.to_datetime(fbi_max_mh.Date)
    fbi_and_rating_max = pd.merge(left=fbi_and_rating_max, right=fbi_max_mh, left_on=['Date', 'Station'], right_on=['Date', 'Station'], how='left')
    
    fbi_and_rating_max.to_csv('./datacube_daily_stats/version_mar25/datacube_aws_location_max_fbi_fdis.csv')
    print("*************************************************************")
    print("In this script, I've ignored a deprecation warning. Please fix at some point")