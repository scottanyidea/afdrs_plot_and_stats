#The previous FBI code calculates spatial moisture, curing and wind to
#calculate FBI.

#Use this code to calculate an alternative FBI, using functions fitted to the 
#GAMs over curing and moisture.

#NOTE for grass condition: 3=natural, 2=grazed, 1=eaten-out

import numpy as np
import pandas as pd
import xarray as xr
from fdrs_calcs import spread_models
from datetime import datetime
import geopandas

M_PER_KM = 1000

def calculate_calibrated_ros(wind, FMC, curing, moist_a0, moist_b0, curing_a1, curing_b1, curing_y0, fuel_condition=2):
    
    moist_curing_factor = calc_moist_curing_factor(FMC, curing, moist_a0, moist_b0, curing_a1, curing_b1, curing_y0)
    
    wind_mask = wind<5
    ROS = np.full(wind.shape, np.nan)
    
    #Equations calculate ROS in km/h
    if fuel_condition==3:
            ROS[wind_mask] = (0.054+0.269*(wind[wind_mask]))
            ROS[~wind_mask] = (1.4+0.838*np.power((wind[~wind_mask]-5), 0.844))
    elif fuel_condition==2:
            ROS[wind_mask] = (0.054+0.209*wind[wind_mask])
            ROS[~wind_mask] = (1.1+0.715*np.power((wind[~wind_mask]-5), 0.844))
    else:
            ROS[wind_mask] = (0.054+0.1045*wind[wind_mask])
            ROS[~wind_mask] = (0.55+0.357*np.power((wind[~wind_mask]-5), 0.844))

    ROS = M_PER_KM * ROS * moist_curing_factor
    return ROS


   
def calc_moist_curing_factor(FMC,curing, fmc_a0, fmc_b0, curing_a1, curing_b1, curing_y0):
    #rescale to ensure value at 100% curing and 0% MC equal to 1
    #note for first term that exp(0) = 1 as FMC = 0
    scale_factor = (fmc_a0) * curing_a1/(1+np.exp(-curing_b1*(100-curing_y0)))
    
    #calculate curing factor values
    factor_ = (fmc_a0*np.exp(-fmc_b0*FMC)) * (curing_a1/(1+np.exp(-curing_b1*(curing-curing_y0))))/scale_factor   
    return factor_

def calc_grass_calibrated_FBI(wind, FMC, curing, moist_a, moist_b, curing_a, curing_b, curing_x0, fuel_condition=2, fuel_load=4.5, return_ros=True):
    ROS  = calculate_calibrated_ros(wind, FMC, curing, moist_a, moist_b, curing_a, curing_b, curing_x0)
    intensity = spread_models.csiro_grassland.calc_intensity(ROS, fuel_load)
    FBI = spread_models.fire_behaviour_index.grass(intensity)
    if return_ros:
        return ROS, intensity, FBI
    else:
        return FBI
    
if __name__=="__main__":
    #Load incidents data with weather variables attached:
    incidents_in = pd.read_pickle("C:/Users/clark/analysis1/incidents_fmc_data/incidents_filtered_and_fbis_2003-2020.pkl")

    inc_wind = incidents_in['Wind'].values
    inc_moisture = incidents_in['AM60_moisture'].values
    inc_curing = incidents_in['Curing_%'].values
    
    #Find FWD the incident is in; to attach the correct parameters:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    incidents_fwd = geopandas.GeoDataFrame(incidents_in, geometry='point', crs=shp_in.crs)
    shp_in.to_crs("EPSG:7899", inplace=True)
    incidents_fwd.to_crs("EPSG:7899", inplace=True)
    #incidents_fwd = geopandas.tools.sjoin(incidents_fwd, shp_in, how='left', predicate='within')
    incidents_fwd = geopandas.tools.sjoin_nearest(incidents_fwd, shp_in, how='left')
    
    area_names = shp_in['Area_Name'].values    

    fbi_area_arr = np.full(len(incidents_fwd), np.nan)
    fbi_len_arr = np.full(len(incidents_fwd), np.nan)

    for reg in area_names:
        print("Starting "+reg)
        
        params_file = pd.read_csv("C:/Users/clark/analysis1/incidents_fmc_data/function_parameters/"+reg+"/curingplusmoisture_2d_95.csv")
        """
        params_file = pd.read_csv("C:/Users/clark/analysis1/incidents_fmc_data/function_parameters//curingplusmoisture_2d_95.csv")
        """
        area_mask = incidents_fwd['Area_Name']==reg

        #Load files with curve parameters:
        #Start with area by incident:
        fit_to_use = 'Area_incident'
        mst_b0 = params_file[params_file['Data_fit']==fit_to_use]['b0'].values
        mst_a0 = params_file[params_file['Data_fit']==fit_to_use]['a0'].values
        cur_b1 = params_file[params_file['Data_fit']==fit_to_use]['b1'].values
        cur_a1 = params_file[params_file['Data_fit']==fit_to_use]['a1'].values
        cur_y0 = params_file[params_file['Data_fit']==fit_to_use]['y0'].values
    
        ROS_calibrated, I_calibrated, FBI_calibrated = calc_grass_calibrated_FBI(
            inc_wind, inc_moisture, inc_curing, mst_a0, mst_b0, cur_a1, cur_b1, cur_y0
            )
        
        fbi_area_arr[area_mask] = FBI_calibrated[area_mask]

        #Now do same for length by incident.
        fit_to_use = 'Length_incident'
        mst_b0 = params_file[params_file['Data_fit']==fit_to_use]['b0'].values
        mst_a0 = params_file[params_file['Data_fit']==fit_to_use]['a0'].values
        cur_b1 = params_file[params_file['Data_fit']==fit_to_use]['b1'].values
        cur_a1 = params_file[params_file['Data_fit']==fit_to_use]['a1'].values
        cur_y0 = params_file[params_file['Data_fit']==fit_to_use]['y0'].values
    
        ROS_calibrated, I_calibrated, FBI_calibrated = calc_grass_calibrated_FBI(
            inc_wind, inc_moisture, inc_curing, mst_a0, mst_b0, cur_a1, cur_b1, cur_y0
            )
        
        fbi_len_arr[area_mask] = FBI_calibrated[area_mask]
        
    incidents_calib = incidents_in
    incidents_calib['FBI_calib_area'] = fbi_area_arr
    incidents_calib['FBI_calib_length'] = fbi_len_arr
    incidents_calib = incidents_calib.rename(columns={'FBI_grazed': 'FBI_orig'})
    
    incidents_calib.to_pickle('./incidents_fmc_data/incidents_calibrated_fbis/incidents_filtered_calibrated2d_fbis_2003-2020_95pc.pkl')
