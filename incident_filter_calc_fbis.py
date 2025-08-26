#For the incident database - filter incidents we want,
#grab the moisture, curing and wind speed, and save for GAM analysis.

#Update 8/1/25: Include ability to calc wind at time of incident too.

#Update 18/6/25: Got everything needed to calc FBI and GFDI, so do that.

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from shapely.geometry import mapping, Polygon
from datetime import datetime, timedelta
from fdrs_calcs import spread_models

HEAT_YIELD = 18600 #kJ/kg

def haversine(lon1, lat1, lon2, lat2):
    #Formula to calculate the distance between lat and lon points so we can
    #determine points within a 10km radius.
    #Convert to radians:
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    
    #Calculate:
    dlon = lon2-lon1
    dlat=lat2-lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    r = 6371
    return c*r


def nearest_hour_index(time_series, pivot_time):
    #Formula to calculate closest hour in wind data to the incident advised time.
    time_diff = np.abs([time - pivot_time for time in time_series])
    return time_diff.argmin(0)

def calculate_gfdi(temp, rh, wind, curing, fuel_load=4.5):
    #Calculate GFDI using Purton equation
    #Temp in deg C, wind in km/h
    GFDI=np.round(        
            np.exp(
                -1.523
                + 1.027 * np.log(fuel_load)
                - 0.009432 * np.power((100 - curing), 1.536)
                + 0.02764 * temp 
                + 0.6422 * np.power(wind, 0.5) 
                - 0.2205 * np.power(rh, 0.5)
                )
                )
    return GFDI

def calculate_grass_fbi(dead_fuel_moisture, wind, curing, fuel_load=4.5, fuel_condition=2):
    #Calculate ROS, intensity, FBI from grass parameters
    ROS = spread_models.csiro_grassland.calc_rate_of_spread(dead_fuel_moisture, wind, curing, fuel_condition)
    intensity = spread_models.common.calc_fire_intensity(ROS, fuel_load, HEAT_YIELD)
    fbi = spread_models.fire_behaviour_index.grass(intensity)
    return ROS, intensity, fbi
    
if __name__=='__main__':
    #Load the incident data.
    incidents_in = pd.read_pickle("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/Victorian Wildfires Research Dataset/incidents.pkl")    
    
    #Initial filtering:
    #Ensure the incident is occurring on a grass pixel.
    #For each incident: If it is small (say <200 ha, but can modify), is the Point on a grass pixel?
    #If large, and the area exists, does the Area contain more than 50% grass pixels?
    recalc_in = xr.open_dataset("C://Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_mar25/recalc_files/VIC_20171005_recalc.nc")
    fuel_type = recalc_in['fuel_type'][2,:,:]
    fuel_type.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    fuel_type.rio.write_crs("EPSG:4326",inplace=True) 
    fuel_type_flag = []

    grass_area_threshold = 0.5
    minimum_area_for_filter = 200
    
    start_date = datetime(2003,4,1)
    end_date = datetime(2020,6,1)
    
    """
    #pre filter: Anywhere with negative text score is not relevant. Remove.
    incidents_in = incidents_in[incidents_in['text_score']>=0]
    #Also remove "non strucutre" fires - most of these will be e.g. car fires
    incidents_in = incidents_in[incidents_in['incident_type']!='Non Structure']
    """
    #This is for the updated v3 incident database.
    #Anywhere with negative text score is not relevant. Remove.
    incidents_in = incidents_in[incidents_in['spreading_fire_flags']>=0]
    #Incidents need to be grass based:
    incidents_in = incidents_in[incidents_in['fuel_type']=='grass']

    #Also trim to between start and end dates
    incidents_in = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    
    print('Filtering incidents by fuel type')
    for index, row in incidents_in.iterrows():
        if ((row['geometry']!=None) & (row['fire_area_ha']>= minimum_area_for_filter)):
            #Clip fuel type map to the fire area.
            area_polygon = row['geometry']
            #NOTE: Need to put the mapped polygon into a list containing 1 element. No idea why but
            #rioxarray doesn't play nice otherwise...
            clipped_map = fuel_type.rio.clip([mapping(area_polygon)], all_touched=False, drop=False)
            #Count number of pixels that are grass, and count total number of pixels in the fire area.
            npix_grass = np.isin(clipped_map, [3004,3016,3020,3042,3044,3046,3062,3064])
            npix_firearea = clipped_map.count()
            #If the fraction of pixels > some threshold (e.g. 0.5) - it's a grass fire. Else, don't call it that and assign it 0.
            print(row['fire_name']+': '+str(npix_grass.sum()/npix_firearea.values))
            
            if npix_grass.sum()/npix_firearea >= grass_area_threshold:
                fuel_type_flag.append(1)
            else:
                fuel_type_flag.append(0)
                
        elif np.isnan(row['latitude']):
                print('Found empty point at position '+str(index)+', dropping.')
                fuel_type_flag.append(0)
        else:
            ft_at_point = fuel_type.sel(longitude=row['longitude'], latitude=row['latitude'], method='nearest').values
            if np.isin(ft_at_point, [3004,3016,3020,3042,3044,3046,3062,3064]):
                fuel_type_flag.append(1)
            else:
                fuel_type_flag.append(0)
    
    incidents_in['fuel_type_flag'] = fuel_type_flag
    incidents_subset = incidents_in[incidents_in['fuel_type_flag']==1] 

    #%%

    #Next step: For each incident, what is the FMC, wind speed, curing, GFDI, FBI?
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    unique_dates = incidents_subset['reported_date'].unique()
    
    #I should produce columns with UTC time so that we can align the wind correctly. This was an issue before...
    incidents_subset['reported_time_utc'] = pd.to_datetime(incidents_subset['reported_time'])-timedelta(hours=11)
    incidents_subset['reported_date_utc'] = pd.to_datetime(incidents_subset['reported_time_utc'].dt.date)
    
    #The VicClim files are monthly. So, to avoid re-loading each time we go to a new incident/day,
    #we grab the month. If we're in the same month, don't re-load.
    #Below is setting up the selection for the first loop (so it loads the first time)
    month_sel = -1
    
    radius_from_incident = 10   #km from incident from which to calculate the average moisture, 
    
    id_for_moisture = []
    moisture_val = []
    curing_val = []
    wind_val = []
    gfdi_val = []
    grassfbi_val = []
    ros_val = []
    intensity_val = []
    for dt in unique_dates:
        print('Starting '+str(dt))
        #Filter to incidents on the day
#        incidents_sel = incidents_subset[incidents_subset['reported_date']==dt]
        incidents_sel = incidents_subset[incidents_subset['reported_date_utc']==dt]
        #Load weather variables from VicClim:
        if dt.month<10:
            mth_str = '0'+str(dt.month)
        else:
            mth_str = str(dt.month)
        if month_sel!=dt.month:
            print("Loading "+str(dt.year)+str(mth_str))
            #Load new data ONLY if we move to a new month.
            temp_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_TSFC1972-2020/"+str(dt.year)+"/"+mth_str+"/IDV71000_VIC_T_SFC.nc")
            rh_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_RSFC1972-2020/"+str(dt.year)+"/"+mth_str+"/IDV71018_VIC_RH_SFC.nc")
            wind_in = xr.open_dataset("M:/Archived/VicClimV5/WRFV5_WMAG1972-2020/"+str(dt.year)+"/"+mth_str+"/IDV71006_VIC_Wind_Mag_SFC.nc")
            if (dt.year<=2016 | ((dt.year==2017) & (dt.month<=6))):
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/pre2017_curing_for_VicClim_"+str(dt.year)+mth_str+".nc")
            else:
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/mapVictoria_curing_for_VicClim_"+str(dt.year)+mth_str+".nc")

            #Calculating McArthur moisture:
            print("Calculating moisture")
            AM60_moist = 9.58 - 0.205*temp_in['T_SFC'].values + 0.138*rh_in['RH_SFC'].values
            #Find daily minimum of moisture, RH, max of temperature
            #Build arrays first then cycle through each day. Doing this for all points across Vic
            print("Find minimum daily")
            n_days = temp_in.time.shape[0]/24
            AM60_moist_min = np.full((int(n_days), len(temp_in.latitude), len(temp_in.longitude)), np.nan)
            temp_max = np.full((int(n_days), len(temp_in.latitude), len(temp_in.longitude)), np.nan)
            rh_min = np.full((int(n_days), len(temp_in.latitude), len(temp_in.longitude)), np.nan)
            dates_wx = []
            for i in range(0,int(n_days)):
                AM60_moist_min[i,:,:] = np.nanmin(AM60_moist[24*i:24*(i+1),:,:], axis=0)
                temp_max[i,:,:] = np.nanmax(temp_in['T_SFC'][24*i:24*(i+1),:,:].values, axis=0)
                rh_min[i,:,:] = np.nanmin(rh_in['RH_SFC'][24*i:24*(i+1),:,:].values, axis=0)
                dates_wx.append(temp_in['time'].values[24*i])
            
            moist_xarr = xr.DataArray(AM60_moist_min, coords=[dates_wx, temp_in.latitude, temp_in.longitude], dims=['time','latitude','longitude'], name=['fuel_moisture'])
            temp_times = pd.Series(dates_wx).dt.date

            lat_grid, lon_grid = xr.broadcast(temp_in.latitude, temp_in.longitude)
            month_sel = dt.month
        #Annoying way to select the date, but should work
        sel_date_ = dt==pd.to_datetime(temp_times)
        
        #For each incident - grab the average moisture and curing in a 10km radius.
        #And calc FBI and FDI from wind, temp, RH and the above.
        print('Calculating moisture, FBIs for incidents on '+str(dt))
        for j in range(0,len(incidents_sel)):
            distance_from_point = haversine(incidents_sel['longitude'].iloc[j], incidents_sel['latitude'].iloc[j], lon_grid, lat_grid)
            AM60_moist_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, moist_xarr.values[sel_date_,:,:][0,:,:], np.nan))
            curing_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, curing_in['GCI'].values[sel_date_,:,:][0,:,:], np.nan))
#            wind_time_ind = nearest_hour_index(wind_in.time.values, incidents_sel['reported_time'].iloc[j])
            wind_time_ind = nearest_hour_index(wind_in.time.values, incidents_sel['reported_time_utc'].iloc[j])
            #NOTE: Converting wind to kts below!
            wind_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, wind_in['Wind_Mag_SFC'].values[wind_time_ind,:,:], np.nan)) * 1.82  
            maxtemp_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, temp_max[sel_date_, :,:][0,:,:], np.nan))
            minrh_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, rh_min[sel_date_, :,:][0,:,:], np.nan))
            gfdi_avg = calculate_gfdi(maxtemp_avg, minrh_avg, wind_avg, curing_avg)
            ros_avg, intensity_avg, fbi_avg = calculate_grass_fbi(AM60_moist_avg, wind_avg, curing_avg)
            
            id_for_moisture.append(incidents_sel.index[j])
            moisture_val.append(AM60_moist_avg)
            curing_val.append(curing_avg)
            wind_val.append(wind_avg)
            gfdi_val.append(gfdi_avg)
            ros_val.append(ros_avg)
            intensity_val.append(intensity_avg)
            grassfbi_val.append(fbi_avg)
            
    
    moisture_df = pd.DataFrame({'ID':id_for_moisture, 'AM60_moisture':moisture_val, 'Curing_%':curing_val, 'Wind':wind_val, 'GFDI':gfdi_val,
                                'ROS_grazed':ros_val, 'Intensity_kWm':intensity_val, 'FBI_grazed':grassfbi_val})
    incidents_out = pd.merge(incidents_subset, moisture_df, left_index=True, right_on='ID', how='inner')
    incidents_out.to_pickle('incidents_filtered_and_fbis_2003-2020.pkl')


    