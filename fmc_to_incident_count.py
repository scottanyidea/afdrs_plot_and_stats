#Script to match count of incidents in the database to daily FMC.

import numpy as np
import pandas as pd
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
from pygam import LogisticGAM, PoissonGAM, GammaGAM, InvGaussGAM, s, te
from mpl_toolkits import mplot3d

if __name__=="__main__":
    #Load incident database:
    incidents_in = pd.read_pickle("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/Suppression Incident Database/incidents.pkl")    
    incidents_in = incidents_in[incidents_in['incident_type']=="Grass"]
    
    #Trim to timeframe:
    start_date = datetime(2008,4,1)
    end_date = datetime(2020,6,30)
    
    #Set area name:
    area_name = 'Northern Country'
    
    incidents_subset = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    incidents_subset = incidents_subset[['season', 'incident_type', 'reported_time', 'containment_time_hr', 'fire_area_ha', 'latitude', 'longitude', 'point', 'relevant_fire_flags']]
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    
    #Filter to "relevant" fires. 
    incidents_subset = incidents_subset[incidents_subset['relevant_fire_flags']>0]
    
    #Load shapefile for FWDs:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    
    #Spatial join to get FWD that the incident is in:
    incidents_subset = geopandas.GeoDataFrame(incidents_subset, geometry='point', crs=shp_in.crs)
    incidents_subset = geopandas.tools.sjoin(incidents_subset, shp_in, how='left', predicate='within')
    
    #Filter to a FWD:
    #incidents_subset = incidents_subset[incidents_subset['Area_Name']==area_name]
    incidents_subset = incidents_subset[incidents_subset['Area_Name'].isin(['Mallee', 'Wimmera', 'Northern Country', 'South West', 'North East'])]
    
    #Count incidents in the area:
    #incidents_count = incidents_subset.groupby('reported_date')['point'].count()
    incidents_count = incidents_subset.groupby(['reported_date', 'Area_Name'])['point'].count()
    
    #Sum up total burnt area from incidents:
    incidents_total_ha = incidents_subset.groupby(['reported_date', 'Area_Name'])['fire_area_ha'].sum()
    
    #Load moisture data:
    moisture_min_data = pd.read_csv('mcarthur_canemc_grass_min_FWD_2.csv', index_col=0)
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
    curing_data = pd.read_csv('vicclim_avg_curing_200804-20206.csv', index_col=0)
    curing_data.index = pd.to_datetime(curing_data.index)
    curing_data = curing_data[(curing_data.index>=start_date) & (curing_data.index<=end_date)]
    curing_data = curing_data.rename_axis('date').reset_index()
    df_b = pd.melt(curing_data, id_vars='date', var_name='curing_location', value_name='curing_%')
    df_b['curing_location'] = df_b['curing_location'].str.replace('_curing', '')
    
    #Merge curing data onto moisture data:
    df_moisture_curing = pd.merge(left=df_a, right=df_b, how='inner', left_on=['date','region'], right_on=['date', 'curing_location'])
    df_moisture_curing = df_moisture_curing.drop(columns='curing_location')
    #remove Central due to weird dominance in ignitions?
    df_moisture_curing = df_moisture_curing[df_moisture_curing['region']!='Central']
    
    #Merge incident count, fire area onto moisture data:
    #moisture_incident_count = pd.merge(left=moisture_min_data, right=incidents_count, how='left', left_index=True, right_index=True)
    moisture_incident_count = pd.merge(left=df_moisture_curing, right=incidents_count, how='left', left_on=['date','region'], right_on=['reported_date','Area_Name'])
    moisture_incident_count = pd.merge(left=moisture_incident_count, right=incidents_total_ha, how='left', left_on=['date','region'], right_on=['reported_date', 'Area_Name'])
    #If the days don't have an incident count, it merges as nan. Fill those with 0s because there are actually no incidents.
    moisture_incident_count['point'] = moisture_incident_count['point'].fillna(0)
    #Do the same for fire area. No incidents=no area burnt.
    moisture_incident_count['fire_area_ha'] = moisture_incident_count['fire_area_ha'].fillna(0)
    #Set up binary - if there was an incident, set as 1, otherwise 0. For binomial count.
    moisture_incident_count['incidents_on_day'] = np.where(moisture_incident_count['point']>0, 1,0)
    
    #moisture_incident_count = moisture_incident_count[((moisture_incident_count['Mallee_CanEMC_min'] < 15) & (moisture_incident_count['incidents_on_day']==1)) | (moisture_incident_count['Mallee_CanEMC_min'] >=15)]
    
    #moisture_incident_count.to_csv('moisture_vs_incidents_all_bar_gip.csv')
    
    #Fit GAMs:
    """
    gam = LogisticGAM().fit(moisture_incident_count[area_name+'_AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    gam_2 = LogisticGAM().fit(moisture_incident_count[area_name+'_CanEMC_min'].values, moisture_incident_count['incidents_on_day'].values)
    
    gam = LogisticGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    gam_2 = LogisticGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['CanEMC_min'].values, moisture_incident_count['incidents_on_day'].values)
    
    #Ok maybe try a different distribution?
#    gam_3 = PoissonGAM().fit(moisture_incident_count[area_name+'_AM60_min'].values, moisture_incident_count['point'].values)
    gam_3 = PoissonGAM().fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values)
    """
    gam_1a = LogisticGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values)
    
    #plot?
    fig, axs = plt.subplots(1)
    xx = gam_1a.generate_X_grid(term=0)
    axs.plot(xx, gam_1a.predict_mu(xx))
#    axs.plot(xx, gam.partial_dependence(term=0, X=xx))
    axs.scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
#    axs.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
#    axs.set_title('Canadian EMC fuel moisture - All except Gippsland')
    axs.set_title('McArthur fuel moisture - All except Gippsland')
    axs.set_xlabel('Estimated fuel moisture')
    axs.set_ylabel("fires/no fires")
    #axs.set_ylabel("number of incidents")
    """
    
    #Maybe we try a 2 factor GAM now!
    gam4 = LogisticGAM(s(0, n_splines=5, spline_order=3)+s(1, n_splines=5, spline_order=3)).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['incidents_on_day'].values)
    gam5 = LogisticGAM(s(0, n_splines=5, spline_order=3)+s(1, n_splines=5, spline_order=3)+te(0,1, n_splines=(5,5))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['incidents_on_day'].values)
    
    #plot again?
    #Start with partial dependence plots
    plt.figure()
    fig, axs = plt.subplots(1,2)
    for i, ax in enumerate(axs):
        xx2 = gam4.generate_X_grid(term=i)
        ax.plot(xx2[:,i], gam5.partial_dependence(term=i, X=xx2))
    
    #OK it's not simple to plot a 3D plot for this. So let's try 2 plots:
    #Hold curing constant at 100%. What happens to moisture?
    #Hold DFMC constant at say 8%. What happens to curing?
    
    fig2, axs2 = plt.subplots(2,1)
    xx3 = gam5.generate_X_grid(term=0)
    xx3[:,1] = 90
    axs2[0].plot(xx3[:,0], gam5.predict_mu(xx3))
    axs2[0].scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    xx4 = gam5.generate_X_grid(term=1)
    xx4[:,0] = 7
    axs2[1].plot(xx4[:,1], gam5.predict_mu(xx4))
    axs2[1].scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    
    
    #OK let's try this way to make a surface plot.
    
    XX = gam5.generate_X_grid(term=2, meshgrid=True)
    Z = gam5.partial_dependence(term=2, X=XX, meshgrid=True)
    fig3, axs3 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs3.plot_surface(XX[0], XX[1], Z, cmap='viridis')
    
    
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

    #Now try total burnt area as a metric. (still using both moisture and curing)
#    gam7 = GammaGAM(s(0, n_splines=5, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['fire_area_ha'].values)
    fig5, axs5 = plt.subplots(1)
#    xx8 = gam7.generate_X_grid(term=0)
#    axs5.plot(xx8, gam7.predict_mu(xx8))
    axs5.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    """