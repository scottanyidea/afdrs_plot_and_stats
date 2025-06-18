#Script to calculate grass fuel moisture from VicClim inputs.
#For now - we calculate the McArthur and Canadian EMC fuel moisture.

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import geopandas
from alt_fuel_moisture_funcs import grass_fuel_moisture
from shapely.geometry import mapping
import warnings
import gc

if __name__=='__main__':
        
    dates_ = pd.date_range(datetime(2003,4,1), datetime(2020,6,30), freq='D')
    
    #Load shapefile for clipping:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA/PID90109_VIC_Boundary_SHP_FWA.shp")
#    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp")
#    areas_list = ['Mallee', 'Wimmera', 'Northern Country', 'North East', 'South West', 'Central', 'North Central']
    areas_list = shp_in['Area_Name'].values
    minima_mcarthur = {area_name+'_AM60_min': [] for area_name in areas_list}
    minima_canemc = {area_name+'_CanEMC_min': [] for area_name in areas_list}
    
    for yr in dates_.year.unique():
        print("Commencing "+str(yr))
        months_in_range = dates_[dates_.year==yr].month.unique()
        for mth in range(1, months_in_range.max()+1):
                #Load input weather files:
                print('Loading '+str(mth))
                if mth<10:
                    mth_str = '0'+str(mth)
                else:
                    mth_str = str(mth)
                temp_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_TSFC1972-2020/"+str(yr)+"/"+mth_str+"/IDV71000_VIC_T_SFC.nc")
                rh_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_RSFC1972-2020/"+str(yr)+"/"+mth_str+"/IDV71018_VIC_RH_SFC.nc")
    
                for area_name in areas_list:
                    area_polygon = shp_in[shp_in['Area_Name']==area_name]
                    #Get clip of temp and RH according to area:
                    temp_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
                    temp_in.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
                    temp_clipped = temp_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                    rh_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
                    rh_in.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
                    rh_clipped = rh_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
    
                    #Calculate McArthur fuel moisture:
                    AM60_moist = 9.58 - 0.205*temp_clipped['T_SFC'].values + 0.138*rh_clipped['RH_SFC'].values
                    MC_0 = AM60_moist[0,:,:]
    
                    #Calculate hourly Canadian hFFMC moisture:
                    EMC_moist = np.full(AM60_moist.shape, np.nan)
                    EMC_moist[0,:,:] = MC_0
                    for i in range(1,temp_in['time'].shape[0]):
                        MC_0 = grass_fuel_moisture.calc_can_hffmc_emc(temp_clipped['T_SFC'].values[i-1,:,:], rh_clipped['RH_SFC'].values[i-1,:,:], MC_0)
                        EMC_moist[i,:,:] = MC_0
    
                    #Calculate hourly minima by region:
                    n_days = temp_in['time'].shape[0]/24
                    with warnings.catch_warnings():
                        #This is to ignore the all-NANS slice warning I get here. I know there are nans, I clipped them out!!!
                        warnings.simplefilter("ignore")
                        for i in range(0, int(n_days)):
                            AM60_min = np.nanmean(np.nanmin(AM60_moist[24*i:24*(i+1),:,:], axis=0))
                            EMC_min = np.nanmean(np.nanmin(EMC_moist[24*i:24*(i+1),:,:], axis=0))
                            minima_mcarthur[area_name+'_AM60_min'].append(AM60_min)
                            minima_canemc[area_name+'_CanEMC_min'].append(EMC_min)
                    
                print(str(yr)+str(mth)+" done.")
                del(AM60_moist)
                del(EMC_moist)
                del(rh_clipped)
                del(temp_clipped)
                del(MC_0)
                temp_in.close()
                rh_in.close()
                gc.collect()

                
    am60_df = pd.DataFrame(minima_mcarthur)
    emc_df = pd.DataFrame(minima_canemc)
    final_df = pd.concat([am60_df, emc_df], axis=1)
    final_df.index = dates_
    final_df.to_csv('./incidents_fmc_data/mcarthur_canemc_grass_min_FWD_2003_2020.csv')
