#Now let's look at FBI and FDI vs incident info directly. 

#V2 uses the pre-filtered and FMC mapped incidents from incident_filter_calc_fbis.py

import numpy as np
import pandas as pd
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from pygam import LogisticGAM, GammaGAM, LinearGAM, ExpectileGAM, PoissonGAM, GAM, s, te
from mpl_toolkits import mplot3d
from shapely.geometry import mapping, Polygon
import rioxarray
import cartopy.crs as ccrs
import seaborn
import geojson
from scipy.stats import expectile
import statsmodels.api as sta
from fmc_to_incident_gams_v2 import goodfit, pseudo_rsq

#%%
if __name__=='__main__':
    
    #FIRST STEP: Load the incident data, preprocess to get FWDs and the correct timespan.
    print("Loading incident data")
    #Set timeframe here:
    start_date = datetime(2003,4,1)
    end_date = datetime(2020,6,30)        

    #Load incident database:
    #incidents_in = pd.read_pickle("C:/Users/clark/analysis1/incidents_fmc_data/incidents_calibrated_fbis/incidents_filtered_calibrated_fbis_2003-2020_80pc.pkl")
    #incidents_in = pd.read_pickle("C:/Users/clark/analysis1/incidents_fmc_data/incidents_calibrated_fbis/incidents_filtered_singlecalibrated2d_fbis_2003-2020_95pc.pkl")
    incidents_in = pd.read_pickle("C:/Users/clark/analysis1/incidents_fmc_data/incidents_calibrated_fbis/incidents_filtered_calibrated2d_fbis_2003-2020_95pc.pkl")
    incidents_in = incidents_in[incidents_in['spreading_fire_flags']>=1]
    incidents_subset = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    
    
    #For incidents >200 ha, we need it to contain a mapped geometry.
    incidents_subset = incidents_subset[(((incidents_subset['geometry']!=None) & (incidents_subset['fire_area_ha']>=200)) | (incidents_subset['fire_area_ha']<200))]
    
    #Load shapefile for FWDs, then join on to get the FWD the incident is in.
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    incidents_subset = geopandas.GeoDataFrame(incidents_subset, geometry='point', crs=shp_in.crs)
    shp_in.to_crs('EPSG:7899', inplace=True)
    incidents_subset.to_crs('EPSG:7899', inplace=True)
    #incidents_subset = geopandas.tools.sjoin(incidents_subset, shp_in, how='left', predicate='within')
    incidents_subset = geopandas.tools.sjoin_nearest(incidents_subset, shp_in, how='left')
    
    #%%
    #Further processing to get count of incidents by FWD vs moisture/curing, etc.
    print("Processing to get incident summaries by FWD")
    #Get list of number of incidents by date and FWD:
    incidents_count = incidents_subset.groupby(['reported_date', 'Area_Name'])['point'].count()
    incidents_total_ha = incidents_subset.groupby(['reported_date', 'Area_Name'])[['fire_area_ha', 'containment_time_hr']].sum()

    #Tidier way: I've now got FBI, wind, moisture and curing all by FWD by day. Load this and join instead.
    fbi_region_data = pd.read_csv("C://Users/clark/analysis1/incidents_fmc_data/fbicalib_grass_max_FWD_2003_2020.csv", index_col=0, parse_dates=True)
    fbi_region_data = fbi_region_data[(fbi_region_data.index >= start_date) & (fbi_region_data.index <= end_date)]
    fbi_region_data = fbi_region_data.rename_axis('date').reset_index()
    
    #Restructure data so we have for each row, date and FWD - columns of FMC, curing, wind and FBI.
    df_c = pd.melt(fbi_region_data, id_vars='date', var_name='value_type', value_name='value')
    split_cols = df_c['value_type'].str.split('_', n=1, expand=True)
    df_c['region'] = split_cols[0]
    df_c['value_type'] = split_cols[1]
    df_c = df_c[['date', 'region','value_type','value']]
    df_c = df_c.pivot(columns='value_type', index=['date', 'region'], values='value').reset_index()
    
    fbi_incident_count = pd.merge(left=df_c, right=incidents_count, how='left', left_on=['date', 'region'], right_on=['reported_date', 'Area_Name'])
    fbi_incident_count = pd.merge(left=fbi_incident_count, right=incidents_total_ha, how='left', left_on=['date','region'], right_on=['reported_date','Area_Name'])
    #If the days don't have an incident count, it merges as nan. Fill those with 0s because there are actually no incidents.
    fbi_incident_count['point'] = fbi_incident_count['point'].fillna(0)
    fbi_incident_count = fbi_incident_count.rename(columns={'point': 'incident_count'})
    #Same for fire area. If there's no incidents, area burnt is 0.
    fbi_incident_count['fire_area_ha'] =  fbi_incident_count['fire_area_ha'].fillna(0)
    fbi_incident_count['containment_time_hr'] =fbi_incident_count['containment_time_hr'].fillna(0)
    fbi_incident_count['incidents_on_day'] = np.where(fbi_incident_count['incident_count']>0, 1,0)
    fbi_incident_count = fbi_incident_count[~fbi_incident_count['Curing_%'].isna()]    
    
    
    #Subset to a specific FWD:
    
    fbi_incident_count = fbi_incident_count[fbi_incident_count['region']=='East Gippsland']
    incidents_subset = incidents_subset[incidents_subset['Area_Name']=='East Gippsland']
    

    #%%
    #Now - plot FBI (or another measure) against incident size. (Check ROS and intensity?)
    #This cell: Do fire area

    spline_number= 8
    expectile_level = 0.95
    
    #Calculate a characteristic length from the fire area (sqrt of the area): 
    incidents_subset['char_length_km'] = np.sqrt(incidents_subset['fire_area_ha'])*0.1
    
    #Fit the GAMs! Do linear and expectile.
    gam_area_fbi_mean = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(incidents_subset['FBI_orig'].values, incidents_subset['fire_area_ha'].values)
    gam_area_fbi95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(incidents_subset['FBI_orig'].values, incidents_subset['fire_area_ha'].values)
    gam_area_calib_mean = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(incidents_subset['FBI_calib_area'].values, incidents_subset['fire_area_ha'].values)
    gam_area_calib95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(incidents_subset['FBI_calib_area'].values, incidents_subset['fire_area_ha'].values)

    #Goodness of fit metrics.
    #First - calculate residuals for prediction and for null (in this case, null is just taking 95th (or nth) expectile)
    area_fbi_res95 = incidents_subset['fire_area_ha'].values - gam_area_fbi95.predict(incidents_subset['FBI_orig'].values)
    area_fbicalib_res95 = incidents_subset['fire_area_ha'].values - gam_area_calib95.predict(incidents_subset['FBI_calib_area'].values)
    area_null95 = incidents_subset['fire_area_ha'].values - expectile(incidents_subset['fire_area_ha'].values, alpha=expectile_level)
    print('*By incident:*')
    fbi_good = goodfit(area_fbi_res95, area_null95, expectile_level)
    fbi_calib_good = goodfit(area_fbicalib_res95, area_null95, expectile_level)
    print('Goodness of fit, area, for original FBI: %1.4f ' % fbi_good)
    print('Goodness of fit, area, for calibrated FBI: %1.4f' % fbi_calib_good)
    
    #Plot these.
    fig_indices, axs_indices = plt.subplots(1,2, figsize=(11,7))
    xx_fbi = gam_area_fbi_mean.generate_X_grid(term=0)
    axs_indices[0].scatter(incidents_subset['FBI_orig'].values, incidents_subset['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs_indices[0].plot(xx_fbi, gam_area_fbi_mean.predict(xx_fbi), color='k')
    axs_indices[0].plot(xx_fbi, gam_area_fbi95.predict(xx_fbi), color='red')    
    axs_indices[0].set_ylabel('Total area burnt (ha)')
    axs_indices[0].set_ylim(0,10000)
    axs_indices[0].set_xlabel('FBI')
    axs_indices[0].set_title('FBI', fontsize=16)
    axs_indices[0].legend(['points','mean', '95%'])
    axs_indices[1].scatter(incidents_subset['FBI_calib_area'].values, incidents_subset['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs_indices[1].plot(xx_fbi, gam_area_calib_mean.predict(xx_fbi), color='k')
    axs_indices[1].plot(xx_fbi, gam_area_calib95.predict(xx_fbi), color='red')    
    axs_indices[1].set_ylabel('Total area burnt (ha)')
    axs_indices[1].set_ylim(0,10000)
    axs_indices[1].set_xlabel('Calibrated FBI')
    axs_indices[1].set_title('Calibrated FBI', fontsize=16)
    axs_indices[1].legend(['points','mean', '95%'])
    fig_indices.suptitle('Area vs FBI')
    
    #%%
    
    #Do the same for characteristic length.
    gam_len_fbi_mean = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(incidents_subset['FBI_orig'].values, incidents_subset['char_length_km'].values)
    gam_len_fbi95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(incidents_subset['FBI_orig'].values, incidents_subset['char_length_km'].values)
    gam_len_calib_mean = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(incidents_subset['FBI_calib_length'].values, incidents_subset['char_length_km'].values)
    gam_len_calib95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(incidents_subset['FBI_calib_length'].values, incidents_subset['char_length_km'].values)

    #Goodness of fit metrics.
    #First - calculate residuals for prediction and for null (in this case, null is just taking 95th (or nth) expectile)
    len_fbi_res95 = incidents_subset['char_length_km'].values - gam_len_fbi95.predict(incidents_subset['FBI_orig'].values)
    len_fbicalib_res95 = incidents_subset['char_length_km'].values - gam_len_calib95.predict(incidents_subset['FBI_calib_length'].values)
    len_null95 = incidents_subset['char_length_km'].values - expectile(incidents_subset['char_length_km'].values, alpha=expectile_level)
    fbi_good = goodfit(len_fbi_res95, len_null95, expectile_level)
    fbi_calib_good = goodfit(len_fbicalib_res95, len_null95, expectile_level)
    print('Goodness of fit, length, for original FBI: %1.4f ' % fbi_good)
    print('Goodness of fit, length, for calibrated FBI: %1.4f ' % fbi_calib_good)

    #Plot these.
    fig_indices2, axs_indices2 = plt.subplots(1,2, figsize=(11,5))
    xx_fbi = gam_area_fbi_mean.generate_X_grid(term=0)
    axs_indices2[0].scatter(incidents_subset['FBI_orig'].values, incidents_subset['char_length_km'].values, facecolor='gray', edgecolors='none', s=8)
    axs_indices2[0].plot(xx_fbi, gam_len_fbi_mean.predict(xx_fbi), color='k')
    axs_indices2[0].plot(xx_fbi, gam_len_fbi95.predict(xx_fbi), color='red')    
    axs_indices2[0].set_ylabel('char length (km)')
    axs_indices2[0].set_ylim(0,10)
    axs_indices2[0].set_xlabel('FBI')
    axs_indices2[0].set_title('FBI', fontsize=16)
    axs_indices2[0].legend(['points','mean', '95%'])
    axs_indices2[1].scatter(incidents_subset['FBI_calib_length'].values, incidents_subset['char_length_km'].values, facecolor='gray', edgecolors='none', s=8)
    axs_indices2[1].plot(xx_fbi, gam_len_calib_mean.predict(xx_fbi), color='k')
    axs_indices2[1].plot(xx_fbi, gam_len_calib95.predict(xx_fbi), color='red')    
    axs_indices2[1].set_ylabel('char_length (km)')
    axs_indices2[1].set_ylim(0,10)
    axs_indices2[1].set_xlabel('Calibrated FBI')
    axs_indices2[1].set_title('Calibrated FBI', fontsize=16)
    axs_indices2[1].legend(['points','mean', '95%'])
    fig_indices2.suptitle('Characteristic length (sqrt area)')
#%%
    
    #Look at number of incidents in the region:
    
    gam_poisson_fbi = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)).fit(fbi_incident_count['FBI_orig'].values, fbi_incident_count['incident_count'].values)
    gam_poisson_calib_fbi = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)).fit(fbi_incident_count['FBI_calib_poisson'].values, fbi_incident_count['incident_count'].values) 
    #Goodness of fit metrics:
    poisson_psr2 = gam_poisson_fbi._estimate_r2(X=fbi_incident_count['FBI_orig'], y=fbi_incident_count['incident_count'].values)['explained_deviance']
    poissoncalib_psr2 = gam_poisson_calib_fbi._estimate_r2(X=fbi_incident_count['FBI_calib_poisson'], y=fbi_incident_count['incident_count'].values)['explained_deviance']
    
    print('*By region:*')
    print('Pseudo R2 of Poisson (count) GAM vs FBI %1.4f ' % poisson_psr2)
    print('Pseudo R2 of Poisson (count) GAM vs calibrated FBI %1.4f ' % poissoncalib_psr2)
    
    #Plotting:
    fig3, axs3 = plt.subplots(1,2, figsize=(11,5))
    xx_fbi3 = gam_poisson_fbi.generate_X_grid(term=0)
    xx_calib3 = gam_poisson_calib_fbi.generate_X_grid(term=0)
    axs3[0].scatter(fbi_incident_count['FBI_orig'].values, fbi_incident_count['incident_count'].values, facecolor='gray', edgecolors='none', s=8)
    axs3[0].plot(xx_fbi3, gam_poisson_fbi.predict(xx_fbi3), color='k')
    axs3[0].set_ylabel('count of incidents')
    axs3[0].set_ylim(0,5)
    axs3[0].set_xlabel('FBI')
    axs3[0].set_title('FBI', fontsize=16)
    axs3[0].legend(['points','mean'])
    axs3[1].scatter(fbi_incident_count['FBI_calib_poisson'].values, fbi_incident_count['incident_count'].values, facecolor='gray', edgecolors='none', s=8)
    axs3[1].plot(xx_calib3, gam_poisson_calib_fbi.predict(xx_calib3), color='k')
    axs3[1].set_ylabel('count of incidents')
    axs3[1].set_ylim(0,5)
    axs3[1].set_xlabel('FBI')
    axs3[1].set_title('FBI', fontsize=16)
    axs3[1].legend(['points','mean'])
    fig3.suptitle("Count of incidents")

#%%
    
    #Look at total area in the region:
        
    gam_area_fbi_reg = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(fbi_incident_count['FBI_orig'].values, fbi_incident_count['fire_area_ha'].values)
    gam_area_fbi_reg95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(fbi_incident_count['FBI_orig'].values, fbi_incident_count['fire_area_ha'].values)
    gam_area_fbicalib_reg = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(fbi_incident_count['FBI_calib_area'].values, fbi_incident_count['fire_area_ha'].values)
    gam_area_fbicalib_reg95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(fbi_incident_count['FBI_calib_area'].values, fbi_incident_count['fire_area_ha'].values)
    
    #Goodness of fit metrics:
    area_fbi_reg_res95 = fbi_incident_count['fire_area_ha'].values - gam_area_fbi_reg95.predict(fbi_incident_count['FBI_orig'].values)
    area_fbicalib_reg_res95 = fbi_incident_count['fire_area_ha'].values - gam_area_fbicalib_reg95.predict(fbi_incident_count['FBI_calib_area'].values)
    area_reg_null95 = fbi_incident_count['fire_area_ha'].values - expectile(fbi_incident_count['fire_area_ha'].values, alpha=expectile_level)
    fbi_good_reg = goodfit(area_fbi_reg_res95, area_reg_null95, expectile_level)
    fbi_good_reg_calib = goodfit(area_fbicalib_reg_res95, area_reg_null95, expectile_level)
    print("Goodness of fit (pseudo R2), area, by region, FBI: %1.4f " % fbi_good_reg)
    print("Goodness of fit (pseudo R2), area, by region, calibrated FBI: %1.4f " % fbi_good_reg_calib)
    
    #Plot:
    fig4, axs4 = plt.subplots(1,2, figsize=(11,6))
    xx_fbi4 = gam_area_fbi_reg.generate_X_grid(term=0)
    xx_calib4 = gam_area_fbicalib_reg.generate_X_grid(term=0)
    axs4[0].scatter(fbi_incident_count['FBI_orig'].values, fbi_incident_count['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs4[0].plot(xx_fbi4, gam_area_fbi_reg.predict(xx_fbi4), color='k')
    axs4[0].plot(xx_fbi4, gam_area_fbi_reg95.predict(xx_fbi4), color='red')
    axs4[0].set_ylabel('Total area in region, ha')
    axs4[0].set_ylim(0,20000)
    axs4[0].set_xlabel('FBI')
    axs4[0].set_title('FBI', fontsize=16)
    axs4[0].legend(['points','mean', '95%'])
    axs4[1].scatter(fbi_incident_count['FBI_calib_area'].values, fbi_incident_count['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs4[1].plot(xx_calib4, gam_area_fbicalib_reg.predict(xx_calib4), color='k')
    axs4[1].plot(xx_calib4, gam_area_fbicalib_reg95.predict(xx_calib4), color='red')
    axs4[1].set_ylabel('Total area in region, ha')
    axs4[1].set_ylim(0,20000)
    axs4[1].set_xlabel('FBI')
    axs4[1].set_title('FBI', fontsize=16)
    axs4[1].legend(['points','mean', '95%'])
    
    
#%%
    #Square root of total area in the region:
    fbi_incident_count['char_length_km'] = np.sqrt(fbi_incident_count['fire_area_ha'].values) * 0.1
    
    gam_len_fbi_reg = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(fbi_incident_count['FBI_orig'].values, fbi_incident_count['char_length_km'].values)
    gam_len_fbi_reg95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(fbi_incident_count['FBI_orig'].values, fbi_incident_count['char_length_km'].values)
    gam_len_fbicalib_reg = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(fbi_incident_count['FBI_calib_length'].values, fbi_incident_count['char_length_km'].values)
    gam_len_fbicalib_reg95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_level).fit(fbi_incident_count['FBI_calib_length'].values, fbi_incident_count['char_length_km'].values)


    #Goodness of fit metrics:
    len_fbi_reg_res95 = fbi_incident_count['char_length_km'].values - gam_len_fbi_reg95.predict(fbi_incident_count['FBI_orig'].values)
    len_fbicalib_reg_res95 = fbi_incident_count['char_length_km'].values - gam_len_fbicalib_reg95.predict(fbi_incident_count['FBI_calib_length'].values)
    len_reg_null95 = fbi_incident_count['char_length_km'].values - expectile(fbi_incident_count['char_length_km'].values, alpha=expectile_level)
    fbi_good_reg = goodfit(len_fbi_reg_res95, len_reg_null95, expectile_level)
    fbi_good_reg_calib = goodfit(len_fbicalib_reg_res95, len_reg_null95, expectile_level)
    print("Goodness of fit (pseudo R2), length, by region, FBI: %1.4f " % fbi_good_reg)
    print("Goodness of fit (pseudo R2), length, by region, calibrated FBI: %1.4f " % fbi_good_reg_calib)

    #Plot:        
    fig5, axs5 = plt.subplots(1,2, figsize=(11,5))
    xx_fbi5 = gam_area_fbi_mean.generate_X_grid(term=0)
    xx_calib5 = gam_len_fbicalib_reg95.generate_X_grid(term=0)
    axs5[0].scatter(fbi_incident_count['FBI_orig'].values, fbi_incident_count['char_length_km'].values, facecolor='gray', edgecolors='none', s=8)
    axs5[0].plot(xx_fbi5, gam_len_fbi_reg.predict(xx_fbi5), color='k')
    axs5[0].plot(xx_fbi5, gam_len_fbi_reg95.predict(xx_fbi5), color='red')
    axs5[0].set_ylabel('Square root of total area, converted to km')
    axs5[0].set_ylim(0,10)
    axs5[0].set_xlabel('FBI')
    axs5[0].set_title('FBI', fontsize=16)
    axs5[0].legend(['points','mean', '95%'])
    axs5[1].scatter(fbi_incident_count['FBI_calib_length'].values, fbi_incident_count['char_length_km'].values, facecolor='gray', edgecolors='none', s=8)
    axs5[1].plot(xx_fbi5, gam_len_fbicalib_reg.predict(xx_calib5), color='k')
    axs5[1].plot(xx_fbi5, gam_len_fbicalib_reg95.predict(xx_calib5), color='red')
    axs5[1].set_ylabel('Square root of total area, converted to km')
    axs5[1].set_ylim(0,10)
    axs5[1].set_xlabel('FBI')
    axs5[1].set_title('Calibrated FBI', fontsize=16)
    axs5[1].legend(['points','mean', '95%'])
    