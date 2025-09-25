#Script to calculate grass fuel moisture and FBI from inputs.
#

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import geopandas
from alt_fuel_moisture_funcs import grass_fuel_moisture
from shapely.geometry import mapping
import warnings
import gc
from grass_curing_for_vicclim_avgs import loadGeoTiff, preprocess_forest_grass_mask, regrid_xr
from fdrs_calcs import spread_models
from incident_calc_2dcalibrated_fbis import calc_grass_calibrated_FBI

GRASS_CONDITION = 2  #1=natural, 2=grazed, 3=eaten-out
GRASS_FUEL_LOAD = 4.5  #t/ha
HEAT_YIELD = 18600 #kJ/kg

if __name__=='__main__':
        
    dates_ = pd.date_range(datetime(2003,4,1), datetime(2020,6,30), freq='D')
    
    #Load shapefile for clipping:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA/PID90109_VIC_Boundary_SHP_FWA.shp")
#    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp")
#    areas_list = ['Mallee', 'Wimmera', 'Northern Country', 'North East', 'South West', 'Central', 'North Central']
    areas_list = shp_in['Area_Name'].values
    minima_mcarthur = {area_name+'_AM60_min': [] for area_name in areas_list}
    curing_fr = {area_name+'_Curing_%': [] for area_name in areas_list}
    wind_avg_fr = {area_name+'_avg_wind_kmh': [] for area_name in areas_list}
    fbi_fr ={area_name+'_FBI_orig': [] for area_name in areas_list}
    fbi_calib_bi_fr = {area_name+'_FBI_calib_bi': [] for area_name in areas_list}
    fbi_calib_ps_fr = {area_name+'_FBI_calib_poisson': [] for area_name in areas_list}
    fbi_calib_area_fr = {area_name+'_FBI_calib_area': [] for area_name in areas_list}
    fbi_calib_length_fr = {area_name+'_FBI_calib_length': [] for area_name in areas_list}

    #Load grass forest mask TIF and process:
    grass_forest_mask_tif_path = 'C://Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/v4_forest0grass1malleeheath1heathland0.tif'
    grass_forest_mask = loadGeoTiff(grass_forest_mask_tif_path, da_var_name='grass_forest', as_Dataset = True)
    grass_forest_mask = preprocess_forest_grass_mask(grass_forest_mask)
    # Re-grid mask to VicClim data
    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/pre2017_curing_for_VicClim_200001.nc")  #random choice, any will do
    grass_forest_mask = regrid_xr(curing_in, grass_forest_mask, method = "nearest")
    curing_in.close()    
    
    mask_switch_check=0
    
    for yr in dates_.year.unique():
        print("Commencing "+str(yr))
        months_in_range = dates_[dates_.year==yr].month.unique()
        for mth in range(months_in_range.min(), months_in_range.max()+1):
                #Load input weather files:
                print('Loading '+str(mth))
                if mth<10:
                    mth_str = '0'+str(mth)
                else:
                    mth_str = str(mth)
                temp_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_TSFC1972-2020/"+str(yr)+"/"+mth_str+"/IDV71000_VIC_T_SFC.nc")
                rh_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_RSFC1972-2020/"+str(yr)+"/"+mth_str+"/IDV71018_VIC_RH_SFC.nc")
                wind_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_WMAG1972-2020/"+str(yr)+"/"+mth_str+"/IDV71006_VIC_Wind_Mag_SFC.nc")
                #Extract DataArray and convert to km/h:
                if (yr<=2016 | ((yr==2017) & (mth<=6))):
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/pre2017_curing_for_VicClim_"+str(yr)+mth_str+".nc")
                else:
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/mapVictoria_curing_for_VicClim_"+str(yr)+mth_str+".nc")
                    #Annoyingly - post 2017 curing maps use "latitude" instead of "lat" so it crashes.
                    #Rename the dimensions when we switch over.    
                    if mask_switch_check==0:
                        grass_forest_mask = grass_forest_mask.rename({'lat': 'latitude', 'lon': 'longitude'})
                        mask_switch_check=1

                #Filter curing by grass forest mask:
                curing_in = xr.where(grass_forest_mask['grass_forest']==1, curing_in['GCI'], np.nan)

                for area_name in areas_list:

                    #Load function parameters for calculating calibrated FBIs:
                    #NOTE: This has been moved inside the loop. I have to load this for every month... but these are tiny files anyway
                    func_params = pd.read_csv("C:/Users/clark/analysis1/incidents_fmc_data/function_parameters/"+area_name+"/curingplusmoisture_2d_95.csv")
                    ps_params = func_params[func_params['Data_fit']=='Poisson']
                    area_params = func_params[func_params['Data_fit']=='Area_region']
                    len_params = func_params[func_params['Data_fit']=='Length_region']

                    area_polygon = shp_in[shp_in['Area_Name']==area_name]
                    #Get clip of temp and RH and wind according to area:
                    temp_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
                    temp_in.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
                    temp_clipped = temp_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                    rh_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
                    rh_in.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
                    rh_clipped = rh_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                    wind_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
                    wind_in.rio.write_crs("EPSG:4326", inplace=True)
                    wind_in_clipped = wind_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                    #Extract wind values and convert to km/h:
                    wind_in_clipped = wind_in_clipped['Wind_Mag_SFC'].values * 1.82
                    if mask_switch_check==0:
                        curing_in.rio.set_spatial_dims(x_dim='lon',y_dim='lat',inplace=True) 
                    else:
                        curing_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) 
                    curing_in.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
                    curing_clipped = curing_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)                    
                    #Convert curing to hourly to match other inputs:
                    curing_clipped_hr = curing_clipped.resample(time='1H').interpolate('zero').values
                    #Annoyingly it doesn't stretch out the last day to match the others.
                    curing_clipped_hr = np.append(curing_clipped_hr, np.repeat(curing_clipped_hr[:,:,-1][:,:,np.newaxis], 23, axis=2), axis=2)
                    curing_clipped_hr = np.transpose(curing_clipped_hr, (2,0,1))                    
                    #Calculate McArthur fuel moisture:
                    AM60_moist = 9.58 - 0.205*temp_clipped['T_SFC'].values + 0.138*rh_clipped['RH_SFC'].values
                    
                    #Calculate FBI:
                    ROS = spread_models.csiro_grassland.calc_rate_of_spread(AM60_moist, wind_in_clipped, curing_clipped_hr, GRASS_CONDITION)
                    intensity = spread_models.common.calc_fire_intensity(ROS, GRASS_FUEL_LOAD, HEAT_YIELD)
                    fbi = spread_models.fire_behaviour_index.grass(intensity)
                    
                    #Calculate calibrated FBI for each of the sets of parameters:
                    #For binomial (fire/no fire):
                    fbi_calib_ps = calc_grass_calibrated_FBI(wind_in_clipped, AM60_moist, curing_clipped_hr, 
                                                             ps_params['a0'].values, ps_params['b0'].values,
                                                             ps_params['a1'].values, ps_params['b1'].values, ps_params['y0'].values, return_ros=False)
                    fbi_calib_area = calc_grass_calibrated_FBI(wind_in_clipped, AM60_moist, curing_clipped_hr, 
                                                             area_params['a0'].values, area_params['b0'].values,
                                                             area_params['a1'].values, area_params['b1'].values, area_params['y0'].values, return_ros=False)
                    fbi_calib_len = calc_grass_calibrated_FBI(wind_in_clipped, AM60_moist, curing_clipped_hr, 
                                                             len_params['a0'].values, len_params['b0'].values,
                                                             len_params['a1'].values, len_params['b1'].values, len_params['y0'].values, return_ros=False)
                    
                    
                    
                    #Calculate, by region, daily min moisture, curing, max FBI and wind at time of max FBI:
                    n_days = temp_in['time'].shape[0]/24
                    with warnings.catch_warnings():
                        #This is to ignore the all-NANS slice warning I get here. I know there are nans, I clipped them out!!!
                        warnings.simplefilter("ignore")
                        for i in range(0, int(n_days)):
                            #Minimum daily FMC
                            AM60_min = np.nanmean(np.nanmin(AM60_moist[24*i:24*(i+1),:,:], axis=0))
                            #Curing:
                            curing_mean = np.nanmean(curing_clipped[:,:,i])
                            #Daily FBI (for max and use arg to get wind):
                            fbi_daily = fbi[24*i:24*(i+1),:,:]
                            fbi_calib_ps_daily = fbi_calib_ps[24*i:24*(i+1),:,:]
                            fbi_calib_area_daily = fbi_calib_area[24*i:24*(i+1),:,:]
                            fbi_calib_len_daily = fbi_calib_len[24*i:24*(i+1),:,:]
                            fbi_max = np.nanmean(np.nanmax(fbi_daily, axis=0))
                            wind_daily = wind_in_clipped[24*i:24*(i+1),:,:]
                            wind_max_args = np.expand_dims(np.argmax(fbi_daily, axis=0), axis=0)
                            wind_at_max = np.nanmean(np.take_along_axis(wind_daily, wind_max_args, axis=0))
                            fbi_ps_max = np.nanmean(np.take_along_axis(fbi_calib_ps_daily, wind_max_args, axis=0))
                            fbi_area_max = np.nanmean(np.take_along_axis(fbi_calib_area_daily, wind_max_args, axis=0))
                            fbi_len_max = np.nanmean(np.take_along_axis(fbi_calib_len_daily, wind_max_args, axis=0))
                            minima_mcarthur[area_name+'_AM60_min'].append(AM60_min)
                            curing_fr[area_name+'_Curing_%'].append(curing_mean)
                            wind_avg_fr[area_name+'_avg_wind_kmh'].append(wind_at_max)
                            fbi_fr[area_name+'_FBI_orig'].append(fbi_max)
                            fbi_calib_ps_fr[area_name+'_FBI_calib_poisson'].append(fbi_ps_max)
                            fbi_calib_area_fr[area_name+'_FBI_calib_area'].append(fbi_area_max)
                            fbi_calib_length_fr[area_name+'_FBI_calib_length'].append(fbi_len_max)
                    
                print(str(yr)+str(mth)+" done.")
                del(AM60_moist)
                del(rh_clipped)
                del(temp_clipped)
                del(curing_clipped)
                del(curing_clipped_hr)
                del(wind_in_clipped)
                del(ROS)
                del(fbi)
                del(fbi_calib_bi)
                del(fbi_calib_ps)
                del(fbi_calib_area)
                del(fbi_calib_len)
                temp_in.close()
                rh_in.close()
                wind_in.close()
                curing_in.close() 
                gc.collect()
                
    am60_df = pd.DataFrame(minima_mcarthur)
    curing_df = pd.DataFrame(curing_fr)
    wind_df = pd.DataFrame(wind_avg_fr)
    fbi_df = pd.DataFrame(fbi_fr)
    fbi_calib_bi_df = pd.DataFrame(fbi_calib_bi_fr)
    fbi_calib_ps_df = pd.DataFrame(fbi_calib_ps_fr)
    fbi_calib_area_df = pd.DataFrame(fbi_calib_area_fr)
    fbi_calib_length_df = pd.DataFrame(fbi_calib_length_fr)
    final_df = pd.concat([am60_df, curing_df, wind_df, fbi_df, fbi_calib_bi_df, fbi_calib_ps_df, fbi_calib_area_df, fbi_calib_length_df], axis=1)
    final_df.index = dates_
    final_df.to_csv('./incidents_fmc_data/fbicalib2d_grass_max_FWD_2003_2020.csv')
