#Script to match count of incidents in the database to daily FMC.

#V2 uses the pre-filtered and FMC mapped incidents from incident_filter_calc_fmc_curing.py

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

def goodfit(resid, resid_null, tau):
    #Function to calculate a goodness of fit (effectively a pseudo R2) for a quantile or expectile regression.
    
    #Sum of deviations for the regression:
    #Intention is for (resid<0) to produce 0 if false and 1 if true.
    #So for elements where residual is positive, multiply by tau.
    #Where negative, multiply by tau-1
    #For tau say 0.9, residuals that are positive get strongly weighted, so attract more of a penalty when
    #they are to be minimised
    V1 = resid*(tau - (resid<0))
    V1 = np.nansum(V1)
    
    #Sum of deviations for the null model:
    V0 = resid_null*(tau - (resid_null<0))
    V0 = np.nansum(V0)
    
    #Explained deviance:
    out_expl = 1-(V1/V0)
    
    return out_expl

def pseudo_rsq(gam_model, gam_null, X, y):
    #Function to calculate pseudo R2 from log likelihoods.
    #Takes arguments of the GAM fit to the model, the null model, and the data X and y.
    #NOTE: The null log likelihood must take only 1s as input for X. 
    
    log_lik_model = gam_model.loglikelihood(X, y)
    log_lik_null = gam_null.loglikelihood(np.full(len(X),1).reshape(-1,1),y)
    
    return 1-(log_lik_model/log_lik_null)

#%%

if __name__=="__main__":
    #FIRST STEP: Load all incident and area data, and merge for first hurdle model.
    
    #Load incident database:
    incidents_in = pd.read_pickle("C:/Users/clark/analysis1/incidents_filtered_with_fmc_curing_2003-2020.pkl")
#    incidents_in = pd.read_pickle("C:/Users/clark/analysis1/incidents_filtered_with_fmc_curing.pkl")
    #Instead - try new filter to count more incidents:
    #Incident type is "grass", and either relevant fire flag >0 or comments include "grass".
    #incidents_in = incidents_in[(incidents_in['incident_type']=="Grass") & ((incidents_in['relevant_fire_flags']>0) | (incidents_in['comments'].str.contains('GRASS')))]

    #Let's try some additional filtering.
#    incidents_in = incidents_in[(incidents_in['spreading_fire_flags']>=1) & (incidents_in['text_score']>=0)]
    incidents_in = incidents_in[(incidents_in['spreading_fire_flags']>=1)]
    
    #Trim to timeframe:
    start_date = datetime(2003,4,1)
    end_date = datetime(2020,6,30)
    
    incidents_subset = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    #incidents_subset = incidents_subset[['season', 'incident_type', 'reported_time', 'containment_time_hr', 'fire_area_ha', 'latitude', 'longitude', 'point', 'geometry', 'relevant_fire_flags', 'AM60_moisture', 'Curing_%', 'Wind']]
    incidents_subset = incidents_subset[['season', 'fuel_type', 'reported_time', 'containment_time_hr', 'fire_area_ha', 'latitude', 'longitude', 'point', 'geometry', 'spreading_fire_flags', 'AM60_moisture', 'Curing_%', 'Wind']]
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    
    #further filtering needed:
    #For incidents >200 ha, we need it to contain a mapped geometry.
    incidents_subset = incidents_subset[(((incidents_subset['geometry']!=None) & (incidents_subset['fire_area_ha']>=200)) | (incidents_subset['fire_area_ha']<200))]
    
    #Load shapefile for FWDs:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    #or for LGAs?
    #shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp")
    
    #Spatial join to get FWD that the incident is in:
    incidents_subset = geopandas.GeoDataFrame(incidents_subset, geometry='point', crs=shp_in.crs)
    incidents_subset = geopandas.tools.sjoin(incidents_subset, shp_in, how='left', predicate='within')
    
    #Step 1: Hurdle model. We are going to look at the district wide moisture and curing. Was there an incident?
    
    #Count incidents in the area:
    #incidents_count = incidents_subset.groupby('reported_date')['point'].count()
    incidents_count = incidents_subset.groupby(['reported_date', 'Area_Name'])['point'].count()
    
    #Sum up total burnt area from incidents:
    incidents_total_ha = incidents_subset.groupby(['reported_date', 'Area_Name'])[['fire_area_ha', 'containment_time_hr']].sum()
    
    #Load moisture data:
    moisture_min_data = pd.read_csv('C://Users/clark/analysis1/incidents_fmc_data/mcarthur_canemc_grass_min_FWD_2003_2020.csv', index_col=0, parse_dates=True)
    #moisture_min_data = pd.read_csv('./incidents_fmc_data/mcarthur_canemc_grass_min_LGA.csv', index_col=0)
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
    curing_data = pd.read_csv('C://Users/clark/analysis1/incidents_fmc_data/vicclim_avg_curing_200304-202006.csv', index_col=0)
    #curing_data = pd.read_csv('./incidents_fmc_data/vicclim_avg_curing_200804-20206_LGA.csv', index_col=0)
    curing_data.index = pd.to_datetime(curing_data.index)
    curing_data = curing_data[(curing_data.index>=start_date) & (curing_data.index<=end_date)]
    curing_data = curing_data.rename_axis('date').reset_index()
    df_b = pd.melt(curing_data, id_vars='date', var_name='curing_location', value_name='curing_%')
    df_b['curing_location'] = df_b['curing_location'].str.replace('_curing', '')
    
    #Merge curing data onto moisture data:
    df_moisture_curing = pd.merge(left=df_a, right=df_b, how='inner', left_on=['date','region'], right_on=['date', 'curing_location'])
    df_moisture_curing = df_moisture_curing.drop(columns='curing_location')

    """
    #Finally load the wind data.
    wind_data = pd.read_csv('C://Users/clark/analysis1/incidents_fmc_data/windavg_FWD_200804-202006.csv', index_col=0)
    wind_data.index = pd.to_datetime(wind_data.index)
    wind_data = wind_data[(wind_data.index>=start_date) & (wind_data.index<=end_date)]
    wind_data = wind_data.rename_axis('date').reset_index()
    #Also need to restructure by region, and time of winds.
    df_c = pd.melt(wind_data, id_vars='date', var_name='wind_location', value_name='wind_speed')
    split_cols = df_c['wind_location'].str.split('_', n=1, expand=True)
    df_c['region'] = split_cols[0]
    df_c['wind_time'] = split_cols[1]
    df_c = df_c[['date', 'region', 'wind_time', 'wind_speed']]
    df_c = df_c.pivot(columns='wind_time', index=['date','region'], values='wind_speed').reset_index()

    df_moisture_curing = pd.merge(left=df_moisture_curing, right=df_c, how='inner', left_on=['date', 'region'], right_on=['date','region'])
    """

    #Merge incident count, fire area onto moisture data:
    #moisture_incident_count = pd.merge(left=moisture_min_data, right=incidents_count, how='left', left_index=True, right_index=True)
    moisture_incident_count = pd.merge(left=df_moisture_curing, right=incidents_count, how='left', left_on=['date','region'], right_on=['reported_date','Area_Name'])
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
    
    """
    #Subset to a specific FWD:
    moisture_incident_count = moisture_incident_count[moisture_incident_count['region']=='East Gippsland']
    incidents_subset = incidents_subset[incidents_subset['Area_Name']=='East Gippsland']
    """
    
    #Take square root of area, to calculate a "characteristic fire length"
    #Area measured in hectares. So sqrt gives us units of 100m, so divide by 10 (multiply by 0.1) to give km
    #as the length.
    incidents_subset['char_length'] = np.sqrt(incidents_subset['fire_area_ha'])*0.1
    moisture_incident_count['char_length'] = np.sqrt(moisture_incident_count['fire_area_ha'])*0.1

    
    #moisture_incident_count.to_csv('./incidents_fmc_data/moisture_curing_vs_incidents_allFWAs_2003_2020_fireflag1.csv')
    #incidents_subset.to_csv('./incidents_fmc_data/moisture_curing_vs_area_2003_2020_fireflag.csv')
    
    """
    Get Annaburroo data as optional part of this plot.
    """
    """
    plot_avg_data = pd.read_csv("C://Users/clark/analysis1/1986_Annaburroo_grassland_weather_data/Plot_Average_Data.csv")
    fmc_anna = plot_avg_data['DFMC.%']
    cur_anna = plot_avg_data['Cure.%']
    """
    fig5 = plt.subplots(1)
    ax = seaborn.scatterplot(x=moisture_incident_count['AM60_min'], y=moisture_incident_count['curing_%'], s=4, color='darkgreen', label='Incident data')
    #ax = seaborn.scatterplot(x=fmc_anna, y=cur_anna, s=12,color='black', label='Annaburroo 1986')
    ax.set_xlabel('Fuel moisture')
    ax.legend()
    
    #%%
    #Now time to fit the binomial hurdle step models.
    spline_number = 7

    #Fit GAMs:
    gam_binomial_moisture = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    gam_binomial_curing = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values)
    #gam_binomial_wind = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['wind_2pm'].values, moisture_incident_count['incidents_on_day'].values)
    null_reg_ = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(np.full(len(moisture_incident_count), 1).reshape(-1,1), moisture_incident_count['incidents_on_day'].values)

    #Calculate pseudo R2 metric for moisture:    
    moist_psr2 = pseudo_rsq(gam_binomial_moisture, null_reg_, moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    print("Pseudo R2 for binomial GAM on moisture: %1.4f " % moist_psr2)
    #Now the same for curing.
    curing_good = pseudo_rsq(gam_binomial_curing,null_reg_, moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values)
    print("Pseudo R2 binomial GAM on curing: %1.4f" % curing_good)
    
    #plot?
    fig, axs = plt.subplots(1)
    xx = gam_binomial_moisture.generate_X_grid(term=0)
    binomial_moisture_predict = gam_binomial_moisture.predict_mu(xx)
    axs.plot(xx, binomial_moisture_predict)
#    axs.plot(xx, gam.partial_dependence(term=0, X=xx))
    axs.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axs.set_title('Predicted fuel moisture')
    axs.set_xlabel('Estimated fuel moisture (%)')
#    axs.set_ylabel("fires/no fires")
    axs.set_ylabel("fires/no fires")

    figa, axsa = plt.subplots(1)    
    xx = gam_binomial_curing.generate_X_grid(term=0)
    binomial_curing_pred = gam_binomial_curing.predict_mu(xx)
    axsa.plot(xx, binomial_curing_pred)
    axsa.scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axsa.set_title('Curing - All')
    axsa.set_xlabel('Curing')
    axsa.set_ylabel("fires/no fires")
    
    #Normalise curing curve and compare to Cruz and Cheney curves.
    y_curing = gam_binomial_curing.predict_mu(xx)
    y_curing_norm = (y_curing-np.min(y_curing))/(np.max(y_curing)-np.min(y_curing))
    cheney_curve = 1.12/(1+59.2*np.exp(-0.124*(xx-50)))
    cruz_curve = 1.036/(1+103.98*np.exp(-0.0996*(xx-20)))
    
    figb, axsb = plt.subplots(1)
    axsb.plot(xx, y_curing_norm, label='Normalised GAM')
    axsb.plot(xx, cheney_curve, label='Cheney')
    axsb.plot(xx, cruz_curve, label='Cruz')
    axsb.set_title('Curing - Normalised GAM, Cheney, Cruz functions')
    axsb.set_xlabel('Curing')
    axsb.set_ylabel("func")
    axsb.legend()
    
    """
    #Finally, wind.
    wind_good = pseudo_rsq(gam_binomial_wind, null_reg_, moisture_incident_count['wind_2pm'].values, moisture_incident_count['incidents_on_day'].values)
    print('Pseudo R2 of binomial GAM on wind: %1.3f' % wind_good)
    
    figb, axsb = plt.subplots(1)
    xx = gam_binomial_wind.generate_X_grid(term=0)
    binomial_wind_pred = gam_binomial_wind.predict_mu(xx)
    axsb.plot(xx, binomial_wind_pred)
    axsb.scatter(moisture_incident_count['wind_2pm'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axsb.set_title('Wind')
    axsb.set_xlabel('Wind (km/h)')
    axsb.set_ylabel("fires/no fires")
    """
    #In this case - construct a 2-predictor GAM.
    """
    gam_binomial_2d = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)
                       +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                       +te(0,1, n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc'))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['incidents_on_day'].values)
    """
    gam_binomial_2d = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)
                       +s(1, n_splines=spline_number, spline_order=3)
                       +te(0,1, n_splines=(spline_number,spline_number))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['incidents_on_day'].values)
    
    #Goodness of fit for 2D binomial GAM:
    binom_2d_good = pseudo_rsq(gam_binomial_2d, null_reg_, moisture_incident_count[['AM60_min', 'curing_%']],moisture_incident_count['incidents_on_day'].values)
    print("Pseudo R2 of binomial 2D GAM: %1.4f" % binom_2d_good)
    
    #Slice the 2 factors for plotting.
    fig2, axs2 = plt.subplots(1,2, figsize=(9,4))
    curing_set_bi = 60   #set curing here
    moisture_set_bi = 6  #set moisture here
    
    #First: Hold curing constant, what is the moisture curve?
    xx3 = gam_binomial_2d.generate_X_grid(term=0)
    xx3[:,1] = curing_set_bi
    axs2[0].plot(xx3[:,0], gam_binomial_2d.predict_mu(xx3))
    axs2[0].scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axs2[0].set_title('Moisture, curing fixed '+str(curing_set_bi))
    #Second:Hold moisture constant, what is the curing response?
    xx4 = gam_binomial_2d.generate_X_grid(term=1)
    xx4[:,0] = moisture_set_bi
    axs2[1].plot(xx4[:,1], gam_binomial_2d.predict_mu(xx4))
    axs2[1].scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axs2[1].set_title('Curing, moisture fixed '+str(moisture_set_bi))
    fig2.suptitle('Fixed value cross sections for incident prob')
    
    
    #OK let's try this way to make a surface plot.
    xx5 = gam_binomial_2d.generate_X_grid(term=0, meshgrid=False)
    yy5 = gam_binomial_2d.generate_X_grid(term=1, meshgrid=False)
    xxx = np.empty([len(xx5), len(yy5)])
    yyy = np.empty([len(xx5), len(yy5)])
    Z = np.empty([len(xx5), len(yy5)])
    
    for i in range(0,len(xx5)):
        xxx[:,i] = xx5[:,0]
        yyy[i,:] = yy5[:,1]
        xx5[:,1] = yy5[i,1]
        Z[:,i] = gam_binomial_2d.predict_mu(xx5)
    
    fig3, axs3 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs3.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs3.set_title('2d binomial GAM - fire/no fire probability')
    
    #In 2D we can compare this to the Cheney and Cruz functions.
    
    moisture_incident_count['cheney_moisture_coeff'] = np.exp(-0.108*moisture_incident_count['AM60_min'])
    moisture_incident_count['cheney_curing_coeff'] = 1.12/(1+59.2*np.exp(-0.124*(moisture_incident_count['curing_%'].values-50)))
    moisture_incident_count['cruz_curing_coeff'] = 1.036/(1+103.98*np.exp(-0.0996*(moisture_incident_count['curing_%'].values-20)))
    
    
    cruz_model = sta.GLM(moisture_incident_count['incidents_on_day'].values, moisture_incident_count[['cheney_moisture_coeff','cruz_curing_coeff']], family=sta.families.Binomial()).fit()
    cheney_model = sta.GLM(moisture_incident_count['incidents_on_day'].values, moisture_incident_count[['cheney_moisture_coeff','cheney_curing_coeff']], family=sta.families.Binomial()).fit()
    
    cruz_pseudor2 = 1-(cruz_model.llf/cruz_model.llnull)
    cheney_pseudor2 = 1-(cheney_model.llf/cheney_model.llnull)
    print('Pseudo R2 of 2D model using Cruz curing model: %1.4f' % cruz_pseudor2)
    print('Pseudo R2 of 2D model using Cheney curing model: %1.4f' % cheney_pseudor2)
    print("*******************************************")
    """
    gam_binomial_3d = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)
                       +s(1, n_splines=spline_number, spline_order=3)
                       +s(2, n_splines=spline_number, spline_order=3)
                       +te(0,1,2, n_splines=(spline_number,spline_number,spline_number))).fit(moisture_incident_count[['AM60_min', 'curing_%','wind_2pm']].values, moisture_incident_count['incidents_on_day'].values)
    
    #Now let's try a surface plot but we need to hold curing constant.
    xx6 = gam_binomial_3d.generate_X_grid(term=0, meshgrid=False)
    yy6 = gam_binomial_3d.generate_X_grid(term=2, meshgrid=False)
    xxx1 = np.empty([len(xx6), len(yy6)])
    www1 = np.empty([len(xx6), len(yy6)])
    Z1 = np.empty([len(xx6), len(yy6)])
    
    xx6[:,1] = curing_set_bi
    yy6[:,1] = curing_set_bi
    
    for i in range(0,len(xx6)):
        xxx1[:,i] = xx6[:,0]
        www1[i,:] = yy6[:,2]
        xx6[:,2] = yy6[i,2]
        Z1[:,i] = gam_binomial_3d.predict_mu(xx6)
    
    fig4, axs4 = plt.subplots(1, subplot_kw={'projection': '3d'})
    axs4.plot_surface(xxx1, www1, Z1, cmap='viridis')
    axs4.set_title('Go/nogo prediction moisture wind, curing at '+str(curing_set_bi))
    """
    #%%
    #Instead of a hurdle of "is there or is there not an incident", why not count the number.
    #these are already separate models, effectively...
    spline_number=7
    
    
    #Fit GAMs:
    gam_poisson_moisture = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values)
#    gam_poisson_curing = PoissonGAM(s(0, n_splines=spline_number, spline_order=3), constraints='monotonic_inc').fit(moisture_incident_count['curing_%'].values, moisture_incident_count['point'].values)
    gam_poisson_curing = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['point'].values)
 #   gam_poisson_wind = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['wind_2pm'].values, moisture_incident_count['point'].values)
    
    #Get values for pseudo R2:
    poisson_moist_psr2 = gam_poisson_moisture._estimate_r2(X=moisture_incident_count['AM60_min'], y=moisture_incident_count['point'].values)['explained_deviance']
    poisson_curing_psr2 = gam_poisson_curing._estimate_r2(X=moisture_incident_count['curing_%'], y=moisture_incident_count['point'].values)['explained_deviance']
    #poisson_wind_psr2 = gam_poisson_wind._estimate_r2(X=moisture_incident_count['wind_2pm'], y=moisture_incident_count['point'].values)['explained_deviance']
    print('Pseudo R2 of Poisson GAM on moisture: %1.4f ' % poisson_moist_psr2)
    print('Pseudo R2 of Poisson GAM on curing: %1.4f ' % poisson_curing_psr2)
    #print('Pseudo R2 of Poisson GAM on wind: %1.4f ' % poisson_wind_psr2)

    
    #plot:        
    figc, axsc = plt.subplots(1)
    xx2 = gam_poisson_moisture.generate_X_grid(term=0)
    poisson_moisture_predict = gam_poisson_moisture.predict_mu(xx2)
    axsc.plot(xx2, poisson_moisture_predict)
    axsc.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
    axsc.set_title('Predicted fuel moisture')
    axsc.set_xlabel('Estimated fuel moisture (%)')
    #    axs.set_ylabel("fires/no fires")
    axsc.set_ylabel("Number of fires")
    axsc.set_ylim(0,5)
    
    #Now the same for curing.
    figb, axsb = plt.subplots(1)    
    xx2 = gam_poisson_curing.generate_X_grid(term=0)
    poisson_curing_pred = gam_poisson_curing.predict_mu(xx2)
    cheney_curve = 1.12/(1+59.2*np.exp(-0.124*(xx2-50)))
    cruz_curve = 1.036/(1+103.98*np.exp(-0.0996*(xx2-20)))
    cheney_curve_p = cheney_curve*np.max(poisson_curing_pred)   #scale Cheney and Cruz curves to the max in the GAM
    cruz_curve_p = cruz_curve*np.max(poisson_curing_pred)    
    axsb.plot(xx2, poisson_curing_pred, label='GAM')
    axsb.plot(xx2, cheney_curve_p, label="Cheney coeff")
    axsb.plot(xx2, cruz_curve_p, label='Cruz coeff')
    axsb.scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
    axsb.set_title('Curing - All')
    axsb.set_xlabel('Curing')
    axsb.set_ylabel("Number of fires")
    axsb.set_ylim(0,1)
    axsb.legend()    
    #2D version:
    gam_poisson_2d = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)
                           +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                           +te(0,1, n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc'))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['point'].values)

    #Goodness of fit for 2D binomial GAM:
    poiss_2d_good = gam_poisson_2d._estimate_r2(X=moisture_incident_count[['AM60_min', 'curing_%']], y=moisture_incident_count['point'])['explained_deviance']
    print("Pseudo R2 of Poisson 2D GAM: %1.4f" % poiss_2d_good)
    print("*******************************************")
    
    
    #OK let's try this way to make a surface plot.
    xx3a = gam_binomial_2d.generate_X_grid(term=0, meshgrid=False)
    yy3a = gam_binomial_2d.generate_X_grid(term=1, meshgrid=False)
    xxx = np.empty([len(xx3a), len(yy3a)])
    yyy = np.empty([len(xx3a), len(yy3a)])
    Z = np.empty([len(xx3a), len(yy3a)])
    
    for i in range(0,len(xx5)):
        xxx[:,i] = xx3a[:,0]
        yyy[i,:] = yy3a[:,1]
        xx3a[:,1] = yy3a[i,1]
        Z[:,i] = gam_poisson_2d.predict_mu(xx3a)
    
    fig4, axs4 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs4.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs4.set_title('2d Poisson GAM - number of incidents per region per day')
 
    
    
    #%%
    #Area model. Where we have an incident, how big do they get?
    #This should be fairly straightforward.

    #First: Plot curing and moisture, and area is the size of the point.
    
    fire_area_incident = incidents_subset['fire_area_ha']
    curing_incident = incidents_subset['Curing_%']
    moisture_incident = incidents_subset['AM60_moisture']
    
    fig5 = plt.subplots(1)
    ax_sp1 = seaborn.scatterplot(x=incidents_subset[incidents_subset['fire_area_ha']<1000]['AM60_moisture'], 
                                 y=incidents_subset[incidents_subset['fire_area_ha']<1000]['Curing_%'], 
                                 s=3, color='darkgreen')

    cmap_o =plt.get_cmap('ocean')
    my_cmap = pltcolors.LinearSegmentedColormap.from_list('ocean_trunc_0_0.8', cmap_o(np.linspace(0.0,0.7,100)))    
    ax_sp1 = seaborn.scatterplot(x=incidents_subset[incidents_subset['fire_area_ha']>=1000]['AM60_moisture'], 
                                 y=incidents_subset[incidents_subset['fire_area_ha']>=1000]['Curing_%'], 
                                 size=incidents_subset[incidents_subset['fire_area_ha']>=1000]['fire_area_ha'], sizes=(3,200), 
                                 hue=incidents_subset[incidents_subset['fire_area_ha']>=1000]['fire_area_ha'], palette=my_cmap)
    
    ax_sp1.set_xlabel('Fuel moisture')
    
    fig5b = plt.subplots(1)
    seaborn.scatterplot(x=incidents_subset['AM60_moisture'], y=incidents_subset['Wind'], size=incidents_subset['fire_area_ha'], sizes=(3,200))
    
    #Now let's fit our GAMs.
    #Fire area:
    spline_number = 8
    expectile_set = 0.95
    
    gam_area_fmc = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident.values, fire_area_incident.values)
    #gam_area_fmc95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=0.5).fit(moisture_incident.values, fire_area_incident.values)
    gam_area_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_dec'), expectile=expectile_set).fit(moisture_incident.values, fire_area_incident.values)
    
    
    """
    #gam_area_cur = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(incidents_subset['Curing_%'].values, fire_area_incident.values)
    #gam_area_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set, constraints='monotonic_inc').fit(incidents_subset['Curing_%'].values, fire_area_incident.values)
    """
    gam_area_cur = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(curing_incident.values, fire_area_incident.values)
    gam_area_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set).fit(curing_incident.values, fire_area_incident.values)

    gam_area_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                                 +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                                 +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%']].values, fire_area_incident.values)
    

    gam_area_2d95_tensoronly = ExpectileGAM(te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%']].values, fire_area_incident.values)
    
    print("***By incident:***")
    area_moist_res95 = fire_area_incident.values - gam_area_fmc95.predict(moisture_incident.values)
    #The null model should probably be the 95th expectile of the data.
    area_null95 = fire_area_incident.values - expectile(fire_area_incident.values, alpha=expectile_set)
    area_moist_good = goodfit(area_moist_res95, area_null95, expectile_set)
    print('Goodness of fit of area GAM on moisture only: %1.3f' % area_moist_good)

    area_cur_res95 = fire_area_incident.values - gam_area_cur95.predict(curing_incident.values)
    area_cur_good = goodfit(area_cur_res95, area_null95, expectile_set)
    print('Goodness of fit of area GAM on curing only: %1.3f' % area_cur_good)

    area_2d_res95 = fire_area_incident.values - gam_area_2d95.predict(incidents_subset[['AM60_moisture','Curing_%']].values)
    area_2d_good = goodfit(area_2d_res95, area_null95, expectile_set)
    print('Goodness of fit of 2D area GAM: %1.3f' % area_2d_good)
    
    area_2dtens_res95 = fire_area_incident.values - gam_area_2d95_tensoronly.predict(incidents_subset[['AM60_moisture','Curing_%']].values)
    area_2dtens_good = goodfit(area_2dtens_res95, area_null95, expectile_set)
    print('Goodness of fit of 2d area GAM with tensor term only: %1.3f' % area_2dtens_good)

    fig6, axs6 = plt.subplots(1,2, figsize=(11,5))
    xx8 = gam_area_fmc.generate_X_grid(term=0)
    xx8_cur = gam_area_cur.generate_X_grid(term=0)
    axs6[0].scatter(moisture_incident.values, fire_area_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs6[0].plot(xx8, gam_area_fmc.predict(xx8), color='k')
    axs6[0].plot(xx8, gam_area_fmc95.predict(xx8), color='red')    
#    axs6[0].plot(xx8, gam_area_fmc95_ci90.predict(xx8), color='orange')
#    axs6[0].plot(xx8, gam_area_fmc95_ci10.predict(xx8), color='orange')
    axs6[0].set_ylabel('Total area burnt (ha)')
    axs6[0].set_ylim(0,4000)
    axs6[0].set_xlabel('Moisture (McArthur) %')
    axs6[0].set_title('Fuel moisture', fontsize=16)
    axs6[0].legend(['points','mean', '95%'])
    axs6[1].scatter(curing_incident.values, fire_area_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs6[1].plot(xx8_cur, gam_area_cur.predict(xx8_cur), color='k')
    axs6[1].plot(xx8_cur, gam_area_cur95.predict(xx8_cur), color='red')    
    axs6[1].set_ylim(0,1000)
    axs6[1].set_xlabel('Curing %')
    axs6[1].set_title('Curing', fontsize=16)
    axs6[1].legend(['points','mean', '95%'])
    fig6.suptitle('Total Area burnt, by incident', fontsize=20)
    
    fig6a, axs6a = plt.subplots(1)
    area_curing_pred = gam_area_cur95.predict(xx8_cur)
    area_curing_norm = (area_curing_pred-np.min(area_curing_pred))/(np.max(area_curing_pred)-np.min(area_curing_pred))
    axs6a.plot(xx8_cur, area_curing_norm, label='AreaGAM_95pctile_norm')
    axs6a.plot(xx8_cur, cheney_curve, label='Cheney')
    axs6a.plot(xx8_cur, cruz_curve, label='Cruz')
    axs6a.set_title('Curing - Normalised area GAM, Cheney, Cruz functions')
    axs6a.set_xlabel('Curing')
    axs6a.set_ylabel("func")
    axs6a.legend()    
    
    
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
    axs10[1].set_ylim(0,2000)
    axs10[1].set_title('curing, constant fuel moisture at '+str(moisture_set))

    """
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
    """
    print("************************************")
    
    #%%
    
    spline_number = 8
    expectile_set = 0.95
    
    #Optional extra (comment in or out): Take square root of area, to calculate a "characteristic fire length"
    #Area measured in hectares. So sqrt gives us units of 100m, so divide by 10 (multiply by 0.1) to give km
    #as the length.
    length_incident = incidents_subset['char_length']
    
    #Now do the same GAMs.
    
    #gam_len_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_dec'), expectile=expectile_set).fit(incidents_subset['AM60_moisture'].values, incidents_subset['char_length'].values)
    gam_len_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set).fit(moisture_incident.values, length_incident.values)
    
    #gam_len_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set, constraints='monotonic_inc').fit(curing_incident.values, length_incident.values)
    gam_len_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set).fit(curing_incident.values, length_incident.values)
    gam_len_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                                 +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                                 +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%']].values, length_incident.values)
    

    gam_len_2d95_tensoronly = ExpectileGAM(te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%']].values, length_incident.values)
    
    #Calculate our goodness of fit measures:
    len_moist_res95 = length_incident.values - gam_len_fmc95.predict(moisture_incident.values)
    len_cur_res95 = length_incident.values - gam_len_cur95.predict(curing_incident.values)
    len_2d_res95 = length_incident.values - gam_len_2d95.predict(incidents_subset[['AM60_moisture', 'Curing_%']].values)
    len_2dtens_res95 = length_incident.values - gam_len_2d95_tensoronly.predict(incidents_subset[['AM60_moisture', 'Curing_%']].values)
    len_null95 = length_incident.values - expectile(length_incident.values, alpha=expectile_set)
    
    
    len_moist_good = goodfit(len_moist_res95, len_null95, expectile_set)
    len_cur_good = goodfit(len_cur_res95, len_null95, expectile_set)
    len_2d_good = goodfit(len_2d_res95, len_null95, expectile_set)
    len_2dtens_good = goodfit(len_2dtens_res95, len_null95, expectile_set)
    print('Goodness of fit of characteristic length GAM on moisture only: %1.3f' % len_moist_good)
    print('Goodness of fit of characteristic length GAM on curing only: %1.3f' % len_cur_good)
    print('Goodness of fit of 2D characteristic length GAM: %1.3f' % len_2d_good)
    print('Goodness of fit of 2D characteristic length GAM tensor only: %1.3f' % len_2dtens_good)
    
    cruz_curing1d = 1.036/(1+103.98*np.exp(-0.0996*(xx8_cur-20)))*gam_len_cur95.predict(100)
    cheney_curing1d = 1.12/(1+59.2*np.exp(-0.124*(xx8_cur-50)))*gam_len_cur95.predict(100)
    
    fig13, axs13 = plt.subplots(1,2, figsize=(11,5))
    axs13[0].scatter(moisture_incident.values, length_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs13[0].plot(xx8, gam_len_fmc95.predict(xx8), color='red')    
    axs13[0].set_ylabel('Length (km)')
    axs13[0].set_ylim(0,5)
    axs13[0].set_xlabel('Moisture (McArthur) %')
    axs13[0].set_title('Fuel moisture', fontsize=16)
    axs13[0].legend(['points','95%GAM'])
    axs13[1].scatter(curing_incident.values, length_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs13[1].plot(xx8_cur, gam_len_cur95.predict(xx8_cur), color='red')    
    axs13[1].plot(xx8_cur, cruz_curing1d, color='blue')
    axs13[1].plot(xx8_cur, cheney_curing1d, color='green')
    axs13[1].set_ylim(0,4)
    axs13[1].set_xlabel('Curing %')
    axs13[1].set_title('Curing', fontsize=16)
#    axs13[1].legend(['points','mean', '95%'])
    axs13[1].legend(['points','95%GAM', 'Cruz', 'Cheney'])
    fig13.suptitle('Characteristic fire length, by incident', fontsize=20)


    res_cheney1d = length_incident - (1.12/(1+59.2*np.exp(-0.124*(curing_incident-50))))*gam_len_cur95.predict(100)
    good_cheney1d = goodfit(res_cheney1d, len_null95, expectile_set)
    print(good_cheney1d)
    xx15 = gam_len_2d95.generate_X_grid(term=0, meshgrid=False)
    yy15 = gam_len_2d95.generate_X_grid(term=1, meshgrid=False)
        
    xxx = np.empty([len(xx9), len(yy9)])
    yyy = np.empty([len(xx9), len(yy9)])
    Z = np.empty([len(xx9), len(yy9)])
    for i in range(0,len(xx9)):
            xxx[:,i] = xx15[:,0]
            yyy[i,:] = yy15[:,1]
            xx15[:,1] = yy15[i,1]
            Z[:,i] = gam_len_2d95.predict(xx15)

    fig14, axs14 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs14.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs14.set_title('Char. length vs curing and fuel moisture')
    
    fig8, axs8 = plt.subplots(1,2, figsize=(8,4))
    axs8[0].plot(xx9[:,0], gam_len_2d95.partial_dependence(term=0, X=xx9))
    axs8[0].set_title('Fuel moisture')
    axs8[0].set_xlabel('Moisture (%)')
    axs8[0].set_ylabel('Partial dependence value')
    axs8[0].set_ylim(-0.4, 0.4)
    axs8[0].set_xlim(0,22)
    axs8[0].hlines(y=0, xmin=-2, xmax=24, color='k')
    axs8[1].plot(yy9[:,1], gam_len_2d95.partial_dependence(term=1, X=yy9))
    axs8[1].set_title('Curing')
    axs8[1].set_xlabel('Curing (%)')
    axs8[1].set_ylim(-0.4, 0.4)
    axs8[1].set_xlim(0,100)
    axs8[1].hlines(y=0, xmin=0, xmax=100, color='k')
    
    xy9 = gam_len_2d95.generate_X_grid(term=2, meshgrid=True)
    fig9, axs9 = plt.subplots(1, subplot_kw={"projection": "3d"})
    xy_dep = gam_len_2d95.partial_dependence(term=2, X=xy9, meshgrid=True)
    axs9.plot_surface(xy9[0], xy9[1], xy_dep, cmap='viridis')
    axs9.set_title('Tensor term partial dependence - length')
    
    xxx = np.empty([len(xx9), len(yy9)])
    yyy = np.empty([len(xx9), len(yy9)])
    Z = np.empty([len(xx9), len(yy9)])
    for i in range(0,len(xx9)):
            xxx[:,i] = xx15[:,0]
            yyy[i,:] = yy15[:,1]
            xx15[:,1] = yy15[i,1]
            Z[:,i] = gam_len_2d95_tensoronly.predict(xx15)

    fig15, axs15 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs15.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs15.set_title('Char. length vs curing and fuel moisture, tensor only')
    
    #Compare to the Cheney/Cruz models.
    max_len = gam_len_2d95.predict([[0,100]]) #max is 0% FMC, 100% curing.
    cheney_pred = np.exp(-0.108*moisture_incident)*(1.12/(1+59.2*np.exp(-0.124*(curing_incident.values-50))))*max_len
    cruz_pred = np.exp(-0.108*moisture_incident)*(1.036/(1+103.98*np.exp(-0.0996*(curing_incident.values-20))))*max_len

    res_cruz = length_incident - cruz_pred
    res_cheney = length_incident - cheney_pred    
    cruz_good = goodfit(res_cruz, len_null95, expectile_set)
    cheney_good = goodfit(res_cheney, len_null95, expectile_set)
    print("Goodness for current grass functions, Cruz curing: %1.3f " % cruz_good)
    print("Goodness for current grass functions, Cheney curing: %1.3f " % cheney_good)
    print("**************************************")
    
    
    #%%
    
    #Optional extra (comment in or out): Take square root of area, to calculate a "characteristic fire length"
    #Area measured in hectares. So sqrt gives us units of 100m, so divide by 10 (multiply by 0.1) to give km
    #as the length.
    
    length_region = moisture_incident_count['char_length']
    moisture_region = moisture_incident_count['AM60_min']
    curing_region = moisture_incident_count['curing_%']
    
    #This version takes the total area burnt in the region and the square root of it. So this is region based, not incident based.
    spline_number = 8
    expectile_set = 0.95
    
    #Now do the same GAMs.
    gam_len_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set).fit(moisture_region.values, length_region.values)

    #gam_len_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set, constraints='monotonic_inc').fit(incidents_subset['Curing_%'].values, incidents_subset['char_length'].values)
    gam_len_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set).fit(curing_region.values, length_region.values)
    gam_len_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                             +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                             +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, length_region.values)


    gam_len_2d95_tensoronly = ExpectileGAM(te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, length_region.values)

    #Calculate our goodness of fit measures:
    len_moist_res95 = length_region.values - gam_len_fmc95.predict(moisture_region.values)
    len_cur_res95 = length_region.values - gam_len_cur95.predict(curing_region.values)
    len_2d_res95 = length_region.values - gam_len_2d95.predict(moisture_incident_count[['AM60_min', 'curing_%']].values)
    len_2dtens_res95 = length_region.values - gam_len_2d95_tensoronly.predict(moisture_incident_count[['AM60_min', 'curing_%']].values)
    len_null95 = length_region.values - expectile(length_region.values, alpha=expectile_set)
    
    
    len_moist_good = goodfit(len_moist_res95, len_null95, expectile_set)
    len_cur_good = goodfit(len_cur_res95, len_null95, expectile_set)
    len_2d_good = goodfit(len_2d_res95, len_null95, expectile_set)
    len_2dtens_good = goodfit(len_2dtens_res95, len_null95, expectile_set)
    print("***By district:***")
    print('Goodness of fit of characteristic length GAM on moisture only: %1.3f' % len_moist_good)
    print('Goodness of fit of characteristic length GAM on curing only: %1.3f' % len_cur_good)
    print('Goodness of fit of 2D characteristic length GAM: %1.3f' % len_2d_good)
    print('Goodness of fit of 2D characteristic length GAM tensor only: %1.3f' % len_2dtens_good)
    
    xx6 = gam_len_fmc95.generate_X_grid(term=0)
    xx6_cur = gam_len_cur95.generate_X_grid(term=0)    
    cruz_curing1d = 1.036/(1+103.98*np.exp(-0.0996*(xx6_cur-20)))*gam_len_cur95.predict(100)
    cheney_curing1d = 1.12/(1+59.2*np.exp(-0.124*(xx6_cur-50)))*gam_len_cur95.predict(100)
    
    fig13, axs13 = plt.subplots(1,2, figsize=(11,5))
#    axs13[0].scatter(incidents_subset['AM60_moisture'].values, incidents_subset['char_length'].values, facecolor='gray', edgecolors='none', s=8)
    axs13[0].scatter(moisture_region.values, length_region.values, facecolor='gray', edgecolors='none', s=8)
    axs13[0].plot(xx6, gam_len_fmc95.predict(xx6), color='red')    
    axs13[0].set_ylabel('Length (km)')
    axs13[0].set_ylim(0,2)
    axs13[0].set_xlabel('Moisture (McArthur) %')
    axs13[0].set_title('Fuel moisture', fontsize=16)
    axs13[0].legend(['points','95%GAM'])
#    axs13[1].scatter(incidents_subset['Curing_%'].values, incidents_subset['char_length'].values, facecolor='gray', edgecolors='none', s=8)
    axs13[1].scatter(curing_region.values, length_region.values, facecolor='gray', edgecolors='none', s=8)
    axs13[1].plot(xx6_cur, gam_len_cur95.predict(xx6_cur), color='red')    
    axs13[1].plot(xx6_cur, cruz_curing1d, color='blue')
    axs13[1].plot(xx6_cur, cheney_curing1d, color='green')
    axs13[1].set_ylim(0,2)
    axs13[1].set_xlabel('Curing %')
    axs13[1].set_title('Curing', fontsize=16)
#    axs13[1].legend(['points','mean', '95%'])
    axs13[1].legend(['points','95%GAM', 'Cruz', 'Cheney'])
    fig13.suptitle('Characteristic fire length, by district total', fontsize=20)

    xx15 = gam_len_2d95.generate_X_grid(term=0, meshgrid=False)
    yy15 = gam_len_2d95.generate_X_grid(term=1, meshgrid=False)
        
    xxx = np.empty([len(xx15), len(yy15)])
    yyy = np.empty([len(xx15), len(yy15)])
    Z = np.empty([len(xx15), len(yy15)])
    for i in range(0,len(xx15)):
            xxx[:,i] = xx15[:,0]
            yyy[i,:] = yy15[:,1]
            xx15[:,1] = yy15[i,1]
            Z[:,i] = gam_len_2d95.predict(xx15)

    fig14, axs14 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs14.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs14.set_title('Char. length vs curing and fuel moisture')
    
    fig8, axs8 = plt.subplots(1,2, figsize=(8,4))
    axs8[0].plot(xx15[:,0], gam_len_2d95.partial_dependence(term=0, X=xx15))
    axs8[0].set_title('Fuel moisture')
    axs8[0].set_xlabel('Moisture (%)')
    axs8[0].set_ylabel('Partial dependence value')
    axs8[0].set_ylim(-0.4, 0.4)
    axs8[0].set_xlim(0,22)
    axs8[0].hlines(y=0, xmin=-2, xmax=24, color='k')
    axs8[1].plot(yy15[:,1], gam_len_2d95.partial_dependence(term=1, X=yy15))
    axs8[1].set_title('Curing')
    axs8[1].set_xlabel('Curing (%)')
    axs8[1].set_ylim(-0.4, 0.4)
    axs8[1].set_xlim(0,100)
    axs8[1].hlines(y=0, xmin=0, xmax=100, color='k')
    
    xy9 = gam_len_2d95.generate_X_grid(term=2, meshgrid=True)
    fig9, axs9 = plt.subplots(1, subplot_kw={"projection": "3d"})
    xy_dep = gam_len_2d95.partial_dependence(term=2, X=xy9, meshgrid=True)
    axs9.plot_surface(xy9[0], xy9[1], xy_dep, cmap='viridis')
    axs9.set_title('Tensor term partial dependence - length')
    
    xxx = np.empty([len(xx15), len(yy15)])
    yyy = np.empty([len(xx15), len(yy15)])
    Z = np.empty([len(xx15), len(yy15)])
    for i in range(0,len(xx15)):
            xxx[:,i] = xx15[:,0]
            yyy[i,:] = yy15[:,1]
            xx15[:,1] = yy15[i,1]
            Z[:,i] = gam_len_2d95_tensoronly.predict(xx15)

    fig15, axs15 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs15.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs15.set_title('Char. length vs curing and fuel moisture, tensor only')
    
    #Compare to the Cheney/Cruz models.
    max_len = gam_len_2d95.predict([[0,100]]) #max is 0% FMC, 100% curing.
    cheney_pred = np.exp(-0.108*incidents_subset['AM60_moisture'])*(1.12/(1+59.2*np.exp(-0.124*(incidents_subset['Curing_%'].values-50))))*max_len
    cruz_pred = np.exp(-0.108*incidents_subset['AM60_moisture'])*(1.036/(1+103.98*np.exp(-0.0996*(incidents_subset['Curing_%'].values-20))))*max_len

    res_cruz = incidents_subset['char_length'] - cruz_pred
    res_cheney = incidents_subset['char_length'] - cheney_pred    
    cruz_good = goodfit(res_cruz, len_null95, expectile_set)
    cheney_good = goodfit(res_cheney, len_null95, expectile_set)
    print("Goodness for current grass functions, Cruz curing: %1.3f " % cruz_good)
    print("Goodness for current grass functions, Cheney curing: %1.3f " % cheney_good)
    print("**************************************")
    
    #%%
    
    #We can do the same thing as above but for area. Total area burnt in the district, on a day.
    #Now let's fit our GAMs.

    fire_area_region = moisture_incident_count['fire_area_ha']

    fig9b = plt.subplots(1)
    seaborn.scatterplot(x=moisture_region.values, y=moisture_incident_count['curing_%'].values, size=moisture_incident_count['fire_area_ha'], sizes=(3,200))
    #Fire area:
    spline_number = 8
    expectile_set = 0.95
    
    #gam_area_fmc95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=0.5).fit(incidents_subset['AM60_moisture'].values, incidents_subset['fire_area_ha'].values)
    gam_area_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_dec'), expectile=expectile_set).fit(moisture_region.values, fire_area_region.values)
    
    
    #gam_area_cur = LinearGAM(s(0, n_splines=spline_number, spline_order=3)).fit(incidents_subset['Curing_%'].values, incidents_subset['fire_area_ha'].values)
    gam_area_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set, constraints='monotonic_inc').fit(curing_region.values, fire_area_region.values)

    gam_area_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                                 +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                                 +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, fire_area_region.values)
    

    
    area_moist_res95 = fire_area_region.values - gam_area_fmc95.predict(moisture_region.values)
    #The null model should probably be the 95th expectile of the data.
    area_null95 = fire_area_region.values - expectile(fire_area_region.values, alpha=expectile_set)
    area_moist_good = goodfit(area_moist_res95, area_null95, expectile_set)
    print('Goodness of fit of area GAM on moisture only: %1.3f' % area_moist_good)

    area_cur_res95 = fire_area_region.values - gam_area_cur95.predict(curing_region.values)
    area_cur_good = goodfit(area_cur_res95, area_null95, expectile_set)
    print('Goodness of fit of area GAM on curing only: %1.3f' % area_cur_good)

    area_2d_res95 = fire_area_region.values - gam_area_2d95.predict(moisture_incident_count[['AM60_min','curing_%']].values)
    area_2d_good = goodfit(area_2d_res95, area_null95, expectile_set)
    print('Goodness of fit of 2D area GAM: %1.3f' % area_2d_good)
    
    fig16, axs16 = plt.subplots(1,2, figsize=(11,5))
    xx18 = gam_area_fmc95.generate_X_grid(term=0)
    xx18_cur = gam_area_cur95.generate_X_grid(term=0)
    axs16[0].scatter(moisture_region.values, fire_area_region.values, facecolor='gray', edgecolors='none', s=8)
    axs16[0].plot(xx18, gam_area_fmc95.predict(xx18), color='red')    
    axs16[0].set_ylabel('Total area burnt (ha)')
    axs16[0].set_ylim(0,2000)
    axs16[0].set_xlabel('Moisture (McArthur) %')
    axs16[0].set_title('Fuel moisture', fontsize=16)
    axs16[0].legend(['points','mean', '95%'])
    axs16[1].scatter(curing_region.values, fire_area_region.values, facecolor='gray', edgecolors='none', s=8)
    axs16[1].plot(xx18_cur, gam_area_cur95.predict(xx18_cur), color='red')    
    axs16[1].set_ylim(0,500)
    axs16[1].set_xlabel('Curing %')
    axs16[1].set_title('Curing', fontsize=16)
    axs16[1].legend(['points','mean', '95%'])
    fig16.suptitle('Total Area burnt, by region', fontsize=20)
    
    fig16a, axs16a = plt.subplots(1)
    area_curing_pred = gam_area_cur95.predict(xx18_cur)
    area_curing_norm = (area_curing_pred-np.min(area_curing_pred))/(np.max(area_curing_pred)-np.min(area_curing_pred))
    axs16a.plot(xx18_cur, area_curing_norm, label='AreaGAM_95pctile_norm')
    axs16a.plot(xx18_cur, cheney_curve, label='Cheney')
    axs16a.plot(xx18_cur, cruz_curve, label='Cruz')
    axs16a.set_title('Curing - Normalised area GAM, Cheney, Cruz functions')
    axs16a.set_xlabel('Curing')
    axs16a.set_ylabel("func")
    axs16a.legend()    
    
    
    xx19 = gam_area_2d95.generate_X_grid(term=0, meshgrid=False)
    yy19 = gam_area_2d95.generate_X_grid(term=1, meshgrid=False)
    
    xxx = np.empty([len(xx19), len(yy19)])
    yyy = np.empty([len(xx19), len(yy19)])
    Z = np.empty([len(xx19), len(yy19)])
    for i in range(0,len(xx19)):
        xxx[:,i] = xx19[:,0]
        yyy[i,:] = yy19[:,1]
        xx19[:,1] = yy19[i,1]
        Z[:,i] = gam_area_2d95.predict(xx19)

    fig17, axs17 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs17.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs17.set_title('Area prediction vs curing and fuel moisture')
    

    
    
    #Hold curing constant at 100%. What happens to moisture?
    #Hold DFMC constant at say 8%. What happens to curing?
    
    fig20, axs20 = plt.subplots(2,1, figsize=(5,8))
    curing_set = 90
    xx19[:,1] = curing_set
    axs20[0].plot(xx19[:,0], gam_area_2d95.predict(xx19))
    axs20[0].set_ylim(0,2000)
    axs20[0].set_title('fuel moisture, constant curing at '+str(curing_set))
    moisture_set = 10
    yy19[:,0] = moisture_set
    axs20[1].plot(yy19[:,1], gam_area_2d95.predict(yy19))
    axs20[1].set_ylim(0,2000)
    axs20[1].set_title('curing, constant fuel moisture at '+str(moisture_set))
    print("************************************")
    

    #%%
    """
    """
    #Let's go back to using time to containment. Does it give us better pseudo R2 for the expectile GAM.
    """
    #Problem is that a lot of incidents don't have containment time. Filter to those that do.    
    incidents_subset_time = incidents_subset[['AM60_moisture', 'Curing_%', 'containment_time_hr']]
    incidents_subset_time.dropna(subset='containment_time_hr', inplace=True)
    
    #Fit GAMs:
    gam_time_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_dec'), expectile=expectile_set).fit(incidents_subset_time['AM60_moisture'].values, incidents_subset_time['containment_time_hr'].values)
    gam_time_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_inc'), expectile=expectile_set).fit(incidents_subset_time['Curing_%'].values, incidents_subset_time['containment_time_hr'].values)
    
    gam_time_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                                 +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                                 +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset_time[['AM60_moisture', 'Curing_%']].values, incidents_subset_time['containment_time_hr'].values)
    
    #Calculate goodness of fit metrics:
    time_moist_res95 = incidents_subset_time['containment_time_hr'].values - gam_time_fmc95.predict(incidents_subset_time['AM60_moisture'].values)
    time_cur_res95 = incidents_subset_time['containment_time_hr'].values - gam_time_cur95.predict(incidents_subset_time['Curing_%'].values)
    time_2d_res95 = incidents_subset_time['containment_time_hr'].values - gam_time_2d95.predict(incidents_subset_time[['AM60_moisture', 'Curing_%']].values)
    time_null95 = incidents_subset_time['containment_time_hr'].values - expectile(incidents_subset_time['containment_time_hr'].values, alpha=expectile_set)

    time_moist_good = goodfit(time_moist_res95, time_null95, expectile_set)
    time_cur_good = goodfit(time_cur_res95, time_null95, expectile_set)
    time_2d_good = goodfit(time_2d_res95, time_null95, expectile_set)
    print("Goodness of fit for containment time GAM on moisture only: %1.3f " % time_moist_good)
    print("Goodness of fit for containment time GAM on curing only: %1.3f " % time_cur_good)
    print("Goodness of fit for 2D containment time GAM: %1.3f " % time_2d_good)
    
    #PLOT:
    xx10 = gam_time_fmc95.generate_X_grid(term=0)
    xx10_cur = gam_time_cur95.generate_X_grid(term=0)
    xx11 = gam_time_2d95.generate_X_grid(term=0)
    yy11 = gam_time_2d95.generate_X_grid(term=1)
    fig16, axs16 = plt.subplots(1,2, figsize=(11,5))
    axs16[0].scatter(incidents_subset_time['AM60_moisture'].values, incidents_subset_time['containment_time_hr'].values, facecolor='gray', edgecolors='none', s=8)
    axs16[0].plot(xx10, gam_time_fmc95.predict(xx10), color='red')    
    axs16[0].set_ylabel('Containment time (hr)')
    axs16[0].set_ylim(0,500)
    axs16[0].set_xlabel('Moisture (McArthur) %')
    axs16[0].set_title('Fuel moisture', fontsize=16)
    axs16[0].legend(['points','95%GAM'])
    axs16[1].scatter(incidents_subset_time['Curing_%'].values, incidents_subset_time['containment_time_hr'].values, facecolor='gray', edgecolors='none', s=8)
    axs16[1].plot(xx10_cur, gam_time_cur95.predict(xx10_cur), color='red')    
    axs16[1].set_ylim(0,200)
    axs16[1].set_xlabel('Curing %')
    axs16[1].set_title('Curing', fontsize=16)
    axs16[1].legend(['points','mean', '95%'])
#    axs16[1].legend(['points','95%GAM', 'Cruz', 'Cheney'])
    fig16.suptitle('Time to containment, by incident', fontsize=20)

    #2D plot:
    xxx = np.empty([len(xx11), len(yy11)])
    yyy = np.empty([len(xx11), len(yy11)])
    Z = np.empty([len(xx11), len(yy11)])
    for i in range(0,len(xx11)):
                xxx[:,i] = xx11[:,0]
                yyy[i,:] = yy11[:,1]
                xx11[:,1] = yy11[i,1]
                Z[:,i] = gam_time_2d95.predict(xx11)

    fig18, axs18 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs18.plot_surface(xxx, yyy, Z, cmap='viridis')
    axs18.set_title('Time to contain vs curing and fuel moisture')
    """    

    #%%
    """
    #Let's also try a measure of rate of spread. Divide characteristic length by time to containment. Could have some
    #issues so see how we go.
    #Multiply by 1000 to get m/hr (we had km/h)
    
    incidents_subset['char_ros'] = (incidents_subset['char_length'].values / incidents_subset['containment_time_hr'])*1000
    #Unfortunately there are incidents with no containment time listed. And more with a containment time really 
    #unfeasibly small. So remove these from the data.
    time_threshold = 1
    incidents_subset = incidents_subset[(~np.isnan(incidents_subset['containment_time_hr'].values)) & (incidents_subset['containment_time_hr']>time_threshold)]
    
    #Same GAMs again.
    gam_ros_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3, constraints='monotonic_dec'), expectile=expectile_set).fit(incidents_subset['AM60_moisture'].values, incidents_subset['char_ros'].values)
    gam_ros_cur95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3), expectile=expectile_set, constraints='monotonic_inc').fit(incidents_subset['Curing_%'].values, incidents_subset['char_ros'].values)

    gam_ros_2d95 = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_dec')
                                    +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                                    +te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%']].values, incidents_subset['char_ros'].values)
       

    gam_ros_2d95_tensoronly = ExpectileGAM(te(0,1,n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc')), expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%']].values, incidents_subset['char_ros'].values)
    
    #Calculate our goodness of fit measures:
    ros_moist_res95 = incidents_subset['char_ros'].values - gam_ros_fmc95.predict(incidents_subset['AM60_moisture'].values)
    ros_cur_res95 = incidents_subset['char_ros'].values - gam_ros_cur95.predict(incidents_subset['Curing_%'].values)
    ros_2d_res95 = incidents_subset['char_ros'].values - gam_ros_2d95.predict(incidents_subset[['AM60_moisture', 'Curing_%']].values)
    ros_2dtens_res95 = incidents_subset['char_ros'].values - gam_ros_2d95_tensoronly.predict(incidents_subset[['AM60_moisture', 'Curing_%']].values)
    ros_null95 = incidents_subset['char_ros'].values - expectile(incidents_subset['char_ros'].values, alpha=expectile_set)
    
    ros_moist_good = goodfit(ros_moist_res95, ros_null95, expectile_set)
    ros_cur_good = goodfit(ros_cur_res95, ros_null95, expectile_set)
    ros_2d_good = goodfit(ros_2d_res95, ros_null95, expectile_set)
    ros_2dtens_good = goodfit(ros_2dtens_res95, ros_null95, expectile_set)
    print('Goodness of fit of characteristic ROS GAM on moisture only: %1.3f' % ros_moist_good)
    print('Goodness of fit of characteristic ROS GAM on curing only: %1.3f' % ros_cur_good)
    print('Goodness of fit of 2D characteristic ROS GAM: %1.3f' % ros_2d_good)
    print('Goodness of fit of 2D characteristic ROS GAM tensor only: %1.3f' % ros_2dtens_good)
    
    fig16, axs16 = plt.subplots(1,2, figsize=(11,5))
    axs16[0].scatter(incidents_subset['AM60_moisture'].values, incidents_subset['char_ros'].values, facecolor='gray', edgecolors='none', s=8)
    axs16[0].plot(xx8, gam_ros_fmc95.predict(xx8), color='red')    
    axs16[0].set_ylabel('ROS (m/h)')
    axs16[0].set_ylim(0,2000)
    axs16[0].set_xlabel('Moisture (McArthur) %')
    axs16[0].set_title('Fuel moisture', fontsize=16)
    axs16[0].legend(['points','mean', '95%'])
    axs16[1].scatter(incidents_subset['Curing_%'].values, incidents_subset['char_ros'].values, facecolor='gray', edgecolors='none', s=8)
    axs16[1].plot(xx8_cur, gam_ros_cur95.predict(xx8_cur), color='red')    
    axs16[1].set_ylim(0,2000)
    axs16[1].set_xlabel('Curing %')
    axs16[1].set_title('Curing', fontsize=16)
    axs16[1].legend(['points','mean', '95%'])
    fig16.suptitle('Characteristic fire ROS, by incident', fontsize=20)

    xx17 = gam_len_2d95.generate_X_grid(term=0, meshgrid=False)
    yy17 = gam_len_2d95.generate_X_grid(term=1, meshgrid=False)
        
    xxx5 = np.empty([len(xx17), len(yy17)])
    yyy5 = np.empty([len(xx17), len(yy17)])
    Z5 = np.empty([len(xx17), len(yy17)])
    for i in range(0,len(xx17)):
            xxx5[:,i] = xx17[:,0]
            yyy5[i,:] = yy17[:,1]
            xx17[:,1] = yy17[i,1]
            Z5[:,i] = gam_ros_2d95.predict(xx17)

    fig17, axs17 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs17.plot_surface(xxx5, yyy5, Z5, cmap='viridis')
    axs17.set_title('Char. ROS vs curing and fuel moisture')
    
    xxx5 = np.empty([len(xx17), len(yy17)])
    yyy5 = np.empty([len(xx9), len(yy9)])
    Z5 = np.empty([len(xx9), len(yy9)])
    for i in range(0,len(xx9)):
            xxx5[:,i] = xx17[:,0]
            yyy5[i,:] = yy17[:,1]
            xx17[:,1] = yy17[i,1]
            Z5[:,i] = gam_ros_2d95_tensoronly.predict(xx17)

    fig18, axs18 = plt.subplots(1, subplot_kw={"projection": "3d"})
    axs18.plot_surface(xxx5, yyy5, Z5, cmap='viridis')
    axs18.set_title('Char. ROS vs curing and fuel moisture, tensor only')
    """
    #%%
    """
    #OOh let's try 3 factor GAM. Include wind!
    #This is the area model.
    spline_number = 8
    expectile_set = 0.98
    gam_area_wind =ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=0.5).fit(incidents_subset['Wind'].values, incidents_subset['fire_area_ha'].values)
    gam_area_wind95 =ExpectileGAM(s(0, n_splines=spline_number, spline_order=3), expectile=expectile_set).fit(incidents_subset['Wind'].values, incidents_subset['fire_area_ha'].values)

    
    area_wind_res95 = incidents_subset['fire_area_ha'] - gam_area_wind95.predict(incidents_subset['Wind'].values)
    area_null95 = incidents_subset['fire_area_ha'].values - expectile(incidents_subset['fire_area_ha'].values, alpha=expectile_set)
    area_wind_good = goodfit(area_wind_res95, area_null95, expectile_set)
    print("Goodness of fit of wind only GAM: %1.3f " % area_wind_good)
    
    fig11, axs11 = plt.subplots(1, figsize=(6,6))
    xx10 = gam_area_wind.generate_X_grid(term=0)
    axs11.scatter(incidents_subset['Wind'].values, incidents_subset['fire_area_ha'].values, facecolor='gray', edgecolors='none', s=8)
    axs11.plot(xx10, gam_area_wind.predict(xx10), color='k')
    axs11.plot(xx10, gam_area_wind95.predict(xx10), color='red')    
    axs11.set_ylabel('Total area burnt (ha)')
    axs11.set_ylim(0,2000)
    axs11.set_xlabel('Wind (km/h)')
    axs11.set_title('Fuel moisture/wind', fontsize=16)
    axs11.legend(['points','mean', '95%'])

    gam_area_3d = ExpectileGAM(s(0, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                               +s(1, n_splines=spline_number, spline_order=3)
                               +s(2, n_splines=spline_number, spline_order=3)
                               +te(0,1,2, n_splines=(spline_number, spline_number, spline_number)),
                               expectile=expectile_set).fit(incidents_subset[['AM60_moisture', 'Curing_%', 'Wind']].values, incidents_subset['fire_area_ha'].values)
    
    area_3d_res95 = incidents_subset['fire_area_ha'] - gam_area_3d.predict(incidents_subset[['AM60_moisture', 'Curing_%', 'Wind']].values)
    area_3d95_good = goodfit(area_3d_res95, area_null95, expectile_set)
    print("Goodness of fit of 3D area GAM (ie. with wind): %1.3f" % area_3d95_good)
    
    #Let's see if we can plot a surface of area vs wind and moisture. Hold curing at say 90%
    xx11 =gam_area_3d.generate_X_grid(term=0, meshgrid=False)
    ww11 = gam_area_3d.generate_X_grid(term=2, meshgrid=False)
    
    #Recall - we are setting up the 2D grids for values of x and y axis.
    #Here it's two 2D grids for moisture and wind that we need to create, as well as the third for the area prediction.
    #The three grids together create a 3D surface plot where points on the surface are (MC, wind, area).
    xxx2 = np.empty([len(xx11), len(ww11)])
    www2 = np.empty([len(xx11), len(ww11)])
    Z2 = np.empty([len(xx11), len(ww11)])
    #Predicting using this GAM needs a grid with 3 columns - for MC, curing, wind. We hold curing constant
    #so set the 2nd column as 90% (or whatever we decide above)
    xx11[:,1] = curing_set
    ww11[:,1] = curing_set
    #For each point in X grid, we assign the MC to the MC grid, wind to the wind grid.
    #Then we assign the appropriate wind in the 3rd column for the ith point, then do the Z (area)
    #prediction. So we are calulating varying moisture, set curing and wind for each iteration.\
    #As we go through each point, we advance the wind speed.
    for i in range(0,len(xx11)):
        xxx2[:, i] = xx11[:,0]
        www2[i,:] = ww11[:,2]
        xx11[:,2] = ww11[i,2]
        Z2[:,i] = gam_area_3d.predict(xx11)
    
    fig12, axs12 = plt.subplots(1, subplot_kw={'projection': '3d'})
    axs12.plot_surface(xxx2, www2, Z2, cmap='viridis')
    axs12.set_title('Area prediction moisture wind, curing at '+str(curing_set))
    
    gam_area_3d_tensoronly = ExpectileGAM(te(0,1,2, n_splines=(spline_number, spline_number, spline_number)), 
                                          expectile=0.95).fit(incidents_subset[['AM60_moisture', 'Curing_%', 'Wind']].values, incidents_subset['fire_area_ha'].values)
    
    area_3d_res95 = incidents_subset['fire_area_ha'] - gam_area_3d_tensoronly.predict(incidents_subset[['AM60_moisture', 'Curing_%', 'Wind']].values)
    area_3d95_good = goodfit(area_3d_res95, area_null95, expectile_set)
    print("Goodness of fit of 3D area GAM (ie. with wind) tensor only : %1.3f" % area_3d95_good)
    
    #For each point in X grid, we assign the MC to the MC grid, wind to the wind grid.
    #Then we assign the appropriate wind in the 3rd column for the ith point, then do the Z (area)
    #prediction. So we are calulating varying moisture, set curing and wind for each iteration.\
    #As we go through each point, we advance the wind speed.
    for i in range(0,len(xx11)):
        xxx2[:, i] = xx11[:,0]
        www2[i,:] = ww11[:,2]
        xx11[:,2] = ww11[i,2]
        Z2[:,i] = gam_area_3d_tensoronly.predict(xx11)
        
    fig12, axs12 = plt.subplots(1, subplot_kw={'projection': '3d'})
    axs12.plot_surface(xxx2, www2, Z2, cmap='viridis')
    axs12.set_title('Area prediction moisture wind tensoronly, curing '+str(curing_set))
    """