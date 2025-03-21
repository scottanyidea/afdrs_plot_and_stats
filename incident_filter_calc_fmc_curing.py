#For the incident database - filter incidents we want,
#grab the moisture, curing and wind speed, and save for GAM analysis.

import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from shapely.geometry import mapping, Polygon
from datetime import datetime

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

if __name__=='__main__':
    #Load the incident data.
    incidents_in = pd.read_pickle("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/Suppression Incident Database/incidents.pkl")    
    
    #Initial filtering:
    #Ensure the incident is occurring on a grass pixel.
    #For each incident: If it is small (say <200 ha, but can modify), is the Point on a grass pixel?
    #If large, and the area exists, does the Area contain more than 50% grass pixels?
    recalc_in = xr.open_dataset("C://Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/VIC_20171005_recalc.nc")
    fuel_type = recalc_in['fuel_type'][2,:,:]
    fuel_type.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    fuel_type.rio.write_crs("EPSG:4326",inplace=True) 
    fuel_type_flag = []

    grass_area_threshold = 0.5
    minimum_area_for_filter = 200
    
    start_date = datetime(2008,4,1)
    end_date = datetime(2020,6,30)
    
    #pre filter: Anywhere with negative text score is not relevant. Remove.
    incidents_in = incidents_in[incidents_in['text_score']>=0]
    #Also remove "non strucutre" fires - most of these will be e.g. car fires
    incidents_in = incidents_in[incidents_in['incident_type']!='Non Structure']
    
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

    #Next step: For each incident, what is the FMC, wind speed and curing?
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    unique_dates = incidents_subset['reported_date'].unique()
    
    #The VicClim files are monthly. So, to avoid re-loading each time we go to a new incident/day,
    #we grab the month. If we're in the same month, don't re-load.
    #Below is setting up the selection for the first loop (so it loads the first time)
    month_sel = -1
    
    radius_from_incident = 10   #km from incident from which to calculate the 
    
    id_for_moisture = []
    moisture_val = []
    curing_val = []
    for dt in unique_dates:
        print('Starting '+str(dt))
        #Filter to incidents on the day
        incidents_sel = incidents_subset[incidents_subset['reported_date']==dt]
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
            if (dt.year<=2016 | ((dt.year==2017) & (dt.month<=6))):
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/pre2017_curing_for_VicClim_"+str(dt.year)+mth_str+".nc")
            else:
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/mapVictoria_curing_for_VicClim_"+str(dt.year)+mth_str+".nc")

            #Calculating McArthur moisture:
            print("calculating moisture")
            AM60_moist = 9.58 - 0.205*temp_in['T_SFC'].values + 0.138*rh_in['RH_SFC'].values
            print("Find minimum daily")
            n_days = temp_in.time.shape[0]/24
            AM60_moist_min = np.full((int(n_days), len(temp_in.latitude), len(temp_in.longitude)), np.nan)
            dates_wx = []
            for i in range(0,int(n_days)):
                AM60_moist_min[i,:,:] = np.nanmin(AM60_moist[24*i:24*(i+1),:,:], axis=0)
                dates_wx.append(temp_in['time'].values[24*i])
            
            moist_xarr = xr.DataArray(AM60_moist_min, coords=[dates_wx, temp_in.latitude, temp_in.longitude], dims=['time','latitude','longitude'], name=['fuel_moisture'])
            temp_times = pd.Series(dates_wx).dt.date

            lat_grid, lon_grid = xr.broadcast(temp_in.latitude, temp_in.longitude)
            month_sel = dt.month
        #Annoying way to select the date, but should work
        sel_date_ = dt==pd.to_datetime(temp_times)
        
        #For each incident - grab the average moisture in a 10km radius.
        print('Calculating moisture for incidents on '+str(dt))
        for j in range(0,len(incidents_sel)):
            distance_from_point = haversine(incidents_sel['longitude'].iloc[j], incidents_sel['latitude'].iloc[j], lon_grid, lat_grid)
            AM60_moist_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, moist_xarr.values[sel_date_,:,:][0,:,:], np.nan))
            curing_avg = np.nanmean(xr.where(distance_from_point<radius_from_incident, curing_in['GCI'].values[sel_date_,:,:][0,:,:], np.nan))
            id_for_moisture.append(incidents_sel.index[j])
            moisture_val.append(AM60_moist_avg)
            curing_val.append(curing_avg)
    
    moisture_df = pd.DataFrame({'ID':id_for_moisture, 'AM60_moisture':moisture_val, 'Curing_%':curing_val})
    incidents_out = pd.merge(incidents_subset, moisture_df, left_index=True, right_on='ID', how='inner')
    incidents_out.to_csv('incidents_filtered_with_fmc_curing.csv')    