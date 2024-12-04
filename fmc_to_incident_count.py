#Script to match count of incidents in the database to daily FMC.

import numpy as np
import pandas as pd
import geopandas
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pygam import LogisticGAM, GammaGAM, LinearGAM, ExpectileGAM, GAM, s, te
from mpl_toolkits import mplot3d
from shapely.geometry import mapping, Polygon
import rioxarray
import cartopy.crs as ccrs
import seaborn
import geojson

if __name__=="__main__":
    #Load incident database:
    incidents_in = pd.read_pickle("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/Suppression Incident Database/incidents.pkl")    
    #Instead - try new filter to count more incidents:
    #Incident type is "grass", and either relevant fire flag >0 or comments include "grass".
    incidents_in = incidents_in[(incidents_in['incident_type']=="Grass") & ((incidents_in['relevant_fire_flags']>0) | (incidents_in['comments'].str.contains('GRASS')))]
    #Exclude incidents with a negative text score (which means it includes words like "unattended", "campfire", "not spreading" etc)
    incidents_in = incidents_in[incidents_in['text_score']>=0]

    """        
    #Filter to "relevant" fires. 
    incidents_in = incidents_in[incidents_in['incident_type']=='Grass']
    incidents_in = incidents_in[incidents_in['relevant_fire_flags']>0]
    """

    #Trim to timeframe:
    start_date = datetime(2008,4,1)
    end_date = datetime(2020,6,30)
    
    incidents_subset = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    incidents_subset = incidents_subset[['season', 'incident_type', 'reported_time', 'containment_time_hr', 'fire_area_ha', 'latitude', 'longitude', 'point', 'geometry', 'relevant_fire_flags']]
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)

        
    #Load shapefile for FWDs:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    #or for LGAs?
    #shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp")
    
    #Spatial join to get FWD that the incident is in:
    incidents_subset = geopandas.GeoDataFrame(incidents_subset, geometry='point', crs=shp_in.crs)
    incidents_subset = geopandas.tools.sjoin(incidents_subset, shp_in, how='left', predicate='within')
    
    #Additional filtering: For large incidents (say >1000 ha), ensure the mapped area covers at least a certain proportion (say 50%)
    #of grassland. To avoid large incidents that are actually forest ones...
    #Firstly - load a fuel type map. Any one will do.
    recalc_in = xr.open_dataset("C://Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/VIC_20171005_recalc.nc")
    fuel_type = recalc_in['fuel_type'][2,:,:]
    fuel_type.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    fuel_type.rio.write_crs("EPSG:4326",inplace=True) 
    #Set up a loop to go through each incident.
    fuel_type_flag = []
    grass_area_threshold = 0.5
    
    """
    #debug:
    incidents_subset_geo = incidents_subset[(incidents_subset['geometry']!=None) & (incidents_subset['fire_area_ha']>500)]
    fig, axs = plt.subplots(1, subplot_kw={'projection': ccrs.PlateCarree()})
    index_testing = recalc_in['index_1'][10,:,:]
    polygon1_ = Polygon([(147.6,-36.3), (147.6, -36.0), (148.0, -36.3)])
    index_testing.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    index_testing.rio.write_crs("EPSG:4326",inplace=True)
    index_testing_clipped = index_testing.rio.clip([mapping(polygon1_)], all_touched=False, drop=False)
#    index_testing_clipped = index_testing.rio.clip([mapping(incidents_subset_geo['geometry'].iloc[1])], all_touched=False)
    index_testing_clipped.plot(ax=axs, transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=True)
    axs.plot(*polygon1_.exterior.xy)
    axs.plot(*incidents_subset_geo['geometry'].iloc[1].exterior.xy)
    axs.set_extent([147.5,148.5, -36.5, -35.5])
    """
    for index, row in incidents_subset.iterrows():
        if ((row['geometry']==None) | (row['fire_area_ha']<500)):
            fuel_type_flag.append(1)    #If there is no geometry - assume it's classified correctly.
        else:
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

    incidents_subset['fuel_type_flag'] = fuel_type_flag
    check_incidents = incidents_subset[incidents_subset['fuel_type_flag']==0]
    incidents_subset = incidents_subset[incidents_subset['fuel_type_flag']==1]


    #Filter to specific FWDs:
    #incidents_subset = incidents_subset[incidents_subset['Area_Name'].isin(['Mallee', 'Wimmera', 'Northern Country', 'South West', 'North East', 'Central', 'North Central'])]
    
    #Count incidents in the area:
    #incidents_count = incidents_subset.groupby('reported_date')['point'].count()
    incidents_count = incidents_subset.groupby(['reported_date', 'Area_Name'])['point'].count()
    
    #Sum up total burnt area from incidents:
    incidents_total_ha = incidents_subset.groupby(['reported_date', 'Area_Name'])[['fire_area_ha', 'containment_time_hr']].sum()
    
    #Load moisture data:
    moisture_min_data = pd.read_csv('./incidents_fmc_data/mcarthur_canemc_grass_min_FWD_3.csv', index_col=0)
    #moisture_min_data = pd.read_csv('./incidents_fmc_data/mcarthur_canemc_grass_min_LGA.csv', index_col=0)
    moisture_min_data.index = pd.to_datetime(moisture_min_data.index)
    moisture_min_data = moisture_min_data[(moisture_min_data.index >= start_date) & (moisture_min_data.index <= end_date)]

    #Restructure data so that we have for each row, date + FWD, columns of McArthur and Canadian moisture.
    moisture_min_data = moisture_min_data.rename_axis('date').reset_index()
    df_a = pd.melt(moisture_min_data, id_vars='date', var_name='moisture_location', value_name='moisture_%')
    split_cols = df_a['moisture_location'].str.split('_', n=1, expand=True)
    df_a['region'] = split_cols[0]
    df_a['moisture_model'] = split_cols[1]
    df_a = df_a[['date', 'region', 'moisture_model', 'moisture_%']]
    df_a = df_a.pivot(columns='moisture_model', index=['date', 'region'], values='moisture_%').reset_index()
    
    #Now load curing data and do the same to it.
    curing_data = pd.read_csv('./incidents_fmc_data/vicclim_avg_curing_200804-20206.csv', index_col=0)
    #curing_data = pd.read_csv('./incidents_fmc_data/vicclim_avg_curing_200804-20206_LGA.csv', index_col=0)
    curing_data.index = pd.to_datetime(curing_data.index)
    curing_data = curing_data[(curing_data.index>=start_date) & (curing_data.index<=end_date)]
    curing_data = curing_data.rename_axis('date').reset_index()
    df_b = pd.melt(curing_data, id_vars='date', var_name='curing_location', value_name='curing_%')
    df_b['curing_location'] = df_b['curing_location'].str.replace('_curing', '')
    
    #Merge curing data onto moisture data:
    df_moisture_curing = pd.merge(left=df_a, right=df_b, how='inner', left_on=['date','region'], right_on=['date', 'curing_location'])
    df_moisture_curing = df_moisture_curing.drop(columns='curing_location')

    #remove Central due to weird dominance in ignitions?
    #df_moisture_curing = df_moisture_curing[df_moisture_curing['region']!='Central']
    
    #Merge incident count, fire area onto moisture data:
    #moisture_incident_count = pd.merge(left=moisture_min_data, right=incidents_count, how='left', left_index=True, right_index=True)
    moisture_incident_count = pd.merge(left=df_moisture_curing, right=incidents_count, how='inner', left_on=['date','region'], right_on=['reported_date','Area_Name'])
    moisture_incident_count = pd.merge(left=moisture_incident_count, right=incidents_total_ha, how='left', left_on=['date','region'], right_on=['reported_date', 'Area_Name'])
    #If the days don't have an incident count, it merges as nan. Fill those with 0s because there are actually no incidents.
    moisture_incident_count['point'] = moisture_incident_count['point'].fillna(0)
    #Do the same for fire area and containment time. No incidents=no area burnt.
    moisture_incident_count['fire_area_ha'] = moisture_incident_count['fire_area_ha'].fillna(0)
    moisture_incident_count['containment_time_hr'] = moisture_incident_count['containment_time_hr'].fillna(0)
    #Set up binary - if there was an incident, set as 1, otherwise 0. For binomial count.
    moisture_incident_count['incidents_on_day'] = np.where(moisture_incident_count['point']>0, 1,0)
    
    #Clean out data with no moisture (because areas are too small)
    moisture_incident_count = moisture_incident_count[~moisture_incident_count['curing_%'].isna()]
    
    #moisture_incident_count.to_csv('./incidents_fmc_data/moisture_vs_incidents_allFWAs_mapfiltered.csv')
    
    #Fit GAMs:
    """
    gam = LogisticGAM().fit(moisture_incident_count[area_name+'_AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    gam_2 = LogisticGAM().fit(moisture_incident_count[area_name+'_CanEMC_min'].values, moisture_incident_count['incidents_on_day'].values)
    """
    gam = LogisticGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
#    gam_2 = LogisticGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['CanEMC_min'].values, moisture_incident_count['incidents_on_day'].values)
    
    #Ok maybe try a different distribution?
#    gam_3 = PoissonGAM().fit(moisture_incident_count[area_name+'_AM60_min'].values, moisture_incident_count['point'].values)
#    gam_3 = PoissonGAM().fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values)
    
    gam_1a = LogisticGAM(s(0, n_splines=5, spline_order=3), constraints='monotonic_inc').fit(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values)
    
    #plot?
    fig, axs = plt.subplots(1)
    xx = gam.generate_X_grid(term=0)
    axs.plot(xx, gam.predict_mu(xx))
#    axs.plot(xx, gam.partial_dependence(term=0, X=xx))
    axs.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
#    axs.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
#    axs.set_title('Canadian EMC fuel moisture - All except Gippsland')
    axs.set_title('McArthur fuel moisture - All')
    axs.set_xlabel('Estimated fuel moisture')
#    axs.set_ylabel("fires/no fires")
    axs.set_ylabel("fires/no fires")

    figa, axsa = plt.subplots(1)
    xx = gam_1a.generate_X_grid(term=0)
    axsa.plot(xx, gam_1a.predict_mu(xx))
    axsa.scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axsa.set_title('McArthur fuel moisture - All')
    axsa.set_xlabel('Curing')
    axsa.set_ylabel("fires/no fires")


    
    #Maybe we try a 2 factor GAM now!
    #gam5 = LogisticGAM(s(0, n_splines=5, spline_order=3)+s(1, n_splines=5, spline_order=3)).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['incidents_on_day'].values)
    gam5 = LogisticGAM(s(0, n_splines=5, spline_order=3)
                       +s(1, n_splines=5, spline_order=3)
                       +te(0,1, n_splines=(5,5))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['incidents_on_day'].values)
    
    """ 
    #plot again?
    #Start with partial dependence plots
    plt.figure()
    fig, axs = plt.subplots(1,2)
    for i, ax in enumerate(axs):
        xx2 = gam5.generate_X_grid(term=i)
        ax.plot(xx2[:,i], gam5.partial_dependence(term=i, X=xx2))
    """
    #OK it's not simple to plot a 3D plot for this. So let's try 2 plots:
    #Hold curing constant at 100%. What happens to moisture?
    #Hold DFMC constant at say 8%. What happens to curing?
    
    fig2, axs2 = plt.subplots(2,1, figsize=(4,9))
    xx3 = gam5.generate_X_grid(term=0)
    xx3[:,1] = 90
    axs2[0].plot(xx3[:,0], gam5.predict_mu(xx3))
    axs2[0].scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axs2[0].set_title('Moisture, curing fixed 90')
    xx4 = gam5.generate_X_grid(term=1)
    xx4[:,0] = 10
    axs2[1].plot(xx4[:,1], gam5.predict_mu(xx4))
    axs2[1].scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axs2[1].set_title('Curing, moisture fixed 10')
    fig2.suptitle('Fixed value cross sections for incident prob')
    """
    
    #OK let's try this way to make a surface plot.
    xx5 = gam5.generate_X_grid(term=0, meshgrid=False)
    yy5 = gam5.generate_X_grid(term=1, meshgrid=False)
    xxx = np.empty([len(xx5), len(yy5)])
    yyy = np.empty([len(xx5), len(yy5)])
    Z = np.empty([len(xx5), len(yy5)])
    
    for i in range(0,len(xx5)):
        xxx[:,i] = xx5[:,0]
        yyy[i,:] = yy5[:,1]
        xx5[:,1] = yy5[i,1]
        Z[:,i] = gam5.predict_mu(xx5)
    
    fig5, axs5 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs5.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs5.set_title('2d binomial GAM - fire/no fire probability')
    
    
    #2 factor Poisson GAM now:
    gam6 = PoissonGAM(s(0, n_splines=4, spline_order=3)+s(1, n_splines=4, spline_order=3)+te(0,1, n_splines=(4,4))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['point'].values)
    gam6.summary()
    
    fig4, axs4 = plt.subplots(1,2, figsize=(12,5))
    xx6 = gam6.generate_X_grid(term=0)
    xx6[:,1] = 90
    axs4[0].plot(xx6[:,0], gam6.predict_mu(xx6))
    axs4[0].scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
    axs4[0].set_title('Fuel moisture (AM60)')
    axs4[0].set_ylabel('Number of fires')
    xx7 = gam6.generate_X_grid(term=1)
    xx7[:,0]  =7
    axs4[1].plot(xx7[:,1], gam6.predict_mu(xx7))
    axs4[1].scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
    axs4[1].set_title('Curing')
    axs4[1].set_ylabel('Number of fires')
    """
    """
    #Now try containment time as a metric. (still using both moisture and curing)
    #Also trying some expectiles here!
    gam_cont_fmc = LinearGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['containment_time_hr'].values)
#    gam_cont_fmc = GAM(s(0, n_splines=5, spline_order=3), distribution='gamma', link='identity').fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['containment_time_hr'].values)
    gam_cont_fmc95 = ExpectileGAM(s(0,n_splines=5, spline_order=3), expectile=0.95).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['containment_time_hr'].values)
    gam_cont_cur = LinearGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['containment_time_hr'].values)
    gam_cont_cur95 = ExpectileGAM(s(0,n_splines=5, spline_order=3), expectile=0.95).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['containment_time_hr'].values)

    fig5, axs5 = plt.subplots(1,2, figsize=(11,5))
    xx8 = gam_cont_fmc.generate_X_grid(term=0)
    xx8_cur = gam_cont_cur.generate_X_grid(term=0)
    axs5[0].scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['containment_time_hr'].values, facecolor='gray', edgecolors='none', s=8)
    axs5[0].plot(xx8, gam_cont_fmc.predict(xx8), color='k')
    axs5[0].plot(xx8, gam_cont_fmc95.predict(xx8), color='red')    
    axs5[0].set_ylabel('Total containment time (hr)')
    axs5[0].set_ylim(0,200)
    axs5[0].set_xlabel('Moisture (McArthur) %')
    axs5[0].set_title('Fuel moisture', fontsize=16)
    axs5[0].legend(['points','mean', '95%'])
    axs5[1].scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['containment_time_hr'].values, facecolor='gray', edgecolors='none', s=8)
    axs5[1].plot(xx8_cur, gam_cont_cur.predict(xx8_cur), color='k')
    axs5[1].plot(xx8_cur, gam_cont_cur95.predict(xx8_cur), color='red')    
    axs5[1].set_ylim(0,200)
    axs5[1].set_xlabel('Curing %')
    axs5[1].set_title('Curing', fontsize=16)
    axs5[1].legend(['points','mean', '95%'])
    fig5.suptitle('Time to containment, by day', fontsize=20)
    """
    #Fire area:
    spline_number = 6
        
#    gam_area_fmc = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['fire_area_ha'].values)
    gam_area_fmc = GammaGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['fire_area_ha'].values)
    gam_area_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_dec'), expectile=0.95).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['fire_area_ha'].values)
#    gam_area_cur = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=0.5).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['fire_area_ha'].values)
    gam_area_cur = GammaGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['fire_area_ha'].values)
    gam_area_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=0.95, constraints='monotonic_inc').fit(moisture_incident_count['curing_%'].values, moisture_incident_count['fire_area_ha'].values)

#    gam_area_2d95 = ExpectileGAM(s(0, n_splines=8, spline_order=3)+s(1, n_splines=8, spline_order=3), expectile=0.9).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['fire_area_ha'].values)
#    gam_area_2d95 = ExpectileGAM(s(0, n_splines=8, spline_order=3)+s(1, n_splines=8, spline_order=3), expectile=0.9).fit(moisture_incident_count[['curing_%', 'AM60_min']].values, moisture_incident_count['fire_area_ha'].values)
    gam_area_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                                 +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                                 +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=0.95).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['fire_area_ha'].values)
    

#    gam_area_2d95_tensoronly = ExpectileGAM(te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=0.95).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['fire_area_ha'].values)
    gam_area_2d95_tensoronly = ExpectileGAM(te(0,1,n_splines=(spline_number,spline_number)), expectile=0.95).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['fire_area_ha'].values)

    """
    gam_area_2d95 = GammaGAM(s(0, n_splines=spline_number, spline_order=3)
                             +s(1, n_splines=spline_number, spline_order=3)
                             +te(0,1,n_splines=(spline_number,spline_number))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['fire_area_ha'].values)
    """
    fig6, axs6 = plt.subplots(1,2, figsize=(11,5))
    xx8 = gam_area_fmc.generate_X_grid(term=0)
    xx8_cur = gam_area_cur.generate_X_grid(term=0)
    axs6[0].scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs6[0].plot(xx8, gam_area_fmc.predict(xx8), color='k')
    axs6[0].plot(xx8, gam_area_fmc95.predict(xx8), color='red')    
    axs6[0].set_ylabel('Total area burnt (ha)')
    axs6[0].set_ylim(0,4000)
    axs6[0].set_xlabel('Moisture (McArthur) %')
    axs6[0].set_title('Fuel moisture', fontsize=16)
    axs6[0].legend(['points','mean', '95%'])
    axs6[1].scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs6[1].plot(xx8_cur, gam_area_cur.predict(xx8_cur), color='k')
    axs6[1].plot(xx8_cur, gam_area_cur95.predict(xx8_cur), color='red')    
    axs6[1].set_ylim(0,4000)
    axs6[1].set_xlabel('Curing %')
    axs6[1].set_title('Curing', fontsize=16)
    axs6[1].legend(['points','mean', '95%'])
    fig6.suptitle('Total Area burnt, by day', fontsize=20)
    
    xx9 = gam_area_2d95.generate_X_grid(term=0, meshgrid=False)
    yy9 = gam_area_2d95.generate_X_grid(term=1, meshgrid=False)
    
    xxx = np.empty([len(xx9), len(yy9)])
    yyy = np.empty([len(xx9), len(yy9)])
    Z = np.empty([len(xx9), len(yy9)])
    for i in range(0,len(xx9)):
        xxx[:,i] = xx9[:,0]
        yyy[i,:] = yy9[:,1]
        xx9[:,1] = yy9[i,1]
        Z[:,i] = gam_area_2d95.predict(xx9)

    fig7, axs7 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs7.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs7.set_title('Area prediction vs curing and fuel moisture')
    
    fig8, axs8 = plt.subplots(1,2)
    axs8[0].plot(xx9[:,0], gam_area_2d95.partial_dependence(term=0, X=xx9))
    axs8[1].plot(yy9[:,1], gam_area_2d95.partial_dependence(term=1, X=yy9))
    
    fig9 = plt.subplots(1)
    seaborn.scatterplot(x=moisture_incident_count['AM60_min'].values, y=moisture_incident_count['curing_%'].values, size=moisture_incident_count['fire_area_ha'], sizes=(3,200))
    
    xy9 = gam_area_2d95.generate_X_grid(term=2, meshgrid=True)
    fig9, axs9 = plt.subplots(1, subplot_kw={"projection": "3d"})
    xy_dep = gam_area_2d95.partial_dependence(term=2, X=xy9, meshgrid=True)
    axs9.plot_surface(xy9[0], xy9[1], xy_dep, cmap='viridis')
    axs9.set_title('Tensor term partial dependence - area')
    
    #Hold curing constant at 100%. What happens to moisture?
    #Hold DFMC constant at say 8%. What happens to curing?
    
    fig10, axs10 = plt.subplots(2,1, figsize=(5,8))
    curing_set = 90
    xx9[:,1] = curing_set
    axs10[0].plot(xx9[:,0], gam_area_2d95.predict(xx9))
    axs10[0].set_ylim(0,2000)
    axs10[0].set_title('fuel moisture, constant curing at '+str(curing_set))
    moisture_set = 10
    yy9[:,0] = moisture_set
    axs10[1].plot(yy9[:,1], gam_area_2d95.predict(yy9))
    axs10[1].set_ylim(0,200)
    axs10[1].set_title('curing, constant fuel moisture at '+str(moisture_set))

    #Plot the tensor only GAM.
    xxx1 = np.empty([len(xx9), len(yy9)])
    yyy1 = np.empty([len(xx9), len(yy9)])
    Z1 = np.empty([len(xx9), len(yy9)])
    for i in range(0,len(xx9)):
        xxx1[:,i] = xx9[:,0]
        yyy1[i,:] = yy9[:,1]
        xx9[:,1] = yy9[i,1]
        Z1[:,i] = gam_area_2d95_tensoronly.predict(xx9)
        
    fig11, axs11 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs11.plot_surface(xxx1, yyy1, Z1, cmap='viridis')
    axs11.set_title('Area prediction (tensor only model)')