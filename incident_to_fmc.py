#Script to match incidents in the database to FMC, and histogram them.

import numpy as np
import pandas as pd
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt

if __name__=="__main__":
    #Load incident database:
    incidents_in = pd.read_pickle("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/Suppression Incident Database/incidents.pkl")    
    incidents_in = incidents_in[incidents_in['incident_type']=="Grass"]
    
    #Trim to timeframe:
    start_date = datetime(2008,4,1)
    end_date = datetime(2020,6,30)
    
    incidents_subset = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    incidents_subset = incidents_subset[['season', 'incident_type', 'reported_time', 'containment_time_hr', 'fire_area_ha', 'latitude', 'longitude', 'point', 'relevant_fire_flags']]
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    
    #Filter to "relevant" fires. 
    incidents_subset = incidents_subset[incidents_subset['relevant_fire_flags']>0]
    
    #Filter to "significant" fires.
    #incidents_subset = incidents_subset[incidents_subset['containment_time_hr']>2.]
    
    #Load moisture data:
    moisture_min_data = pd.read_csv('mcarthur_canemc_grass_min_FWD_2.csv', index_col=0)
    moisture_min_data.index = pd.to_datetime(moisture_min_data.index)
    
    #Load curing data too.
    curing_data = pd.read_csv('vicclim_avg_curing_200804-20206.csv', index_col=0)
    curing_data.index = pd.to_datetime(curing_data.index)
    curing_data = curing_data[(curing_data.index>=start_date) & (curing_data.index<=end_date)]
    curing_data = curing_data.rename_axis('date').reset_index()
    
    #Join moisture data to incidents:
    incidents_subset  = pd.merge(left=incidents_subset, right=moisture_min_data, how='left', left_on='reported_date', right_index=True)
    
    #Load shapefile for FWDs:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    
    #Spatial join to get FWD that the incident is in:
    incidents_subset = geopandas.GeoDataFrame(incidents_subset, geometry='point', crs=shp_in.crs)
    incidents_subset = geopandas.tools.sjoin(incidents_subset, shp_in, how='left', predicate='within')
    
    #Filter to a handful for which we have the moisture data:
    areas_list = ['Mallee', 'Wimmera', 'Northern Country', 'South West', 'North East']
    incidents_subset = incidents_subset[incidents_subset['Area_Name'].isin(areas_list)]

    #Create column that has the moisture for the right region:
    AM60_moisture = np.full(len(incidents_subset), np.nan)
    CanEMC_moisture = np.full(len(incidents_subset), np.nan)
    for areas_ in areas_list:
        AM60_moisture[(incidents_subset['Area_Name']==areas_)] = incidents_subset[incidents_subset['Area_Name']==areas_][areas_+'_AM60_min']
        CanEMC_moisture[(incidents_subset['Area_Name']==areas_)] = incidents_subset[incidents_subset['Area_Name']==areas_][areas_+'_CanEMC_min']
    
    #Now we can histogram it?
    fig = plt.figure()
    plt.hist(AM60_moisture, bins=np.arange(0.,24.,0.5), edgecolor='black')
    plt.title('Current McArthur grass moisture, significant fires')
    fig2 = plt.figure()
    plt.hist(CanEMC_moisture, bins=np.arange(0.,24.,0.5), edgecolor='black')
    plt.title('Canadian EMC grass moisture, significant fires')