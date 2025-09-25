#Script to match count of incidents in the database to daily FMC.

#V3 uses the pre-filtered and FMC mapped incidents from incident_filter_calc_fmc_curing.py

#V3 also, from the GAMs produced, fits a function to try and replicate the shape of the GAM.
#Playing around with percentile levels, splines, variables etc should still use V2 for now.

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
import seaborn
import geojson
from scipy.stats import expectile
import statsmodels.api as sta
from scipy.optimize import curve_fit

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

    #Let's try some additional filtering.
    incidents_in = incidents_in[(incidents_in['spreading_fire_flags']>=1)]
    
    #Trim to timeframe:
    start_date = datetime(2003,4,1)
    end_date = datetime(2020,6,30)
    
    incidents_subset = incidents_in[(incidents_in['reported_time']>=start_date) & (incidents_in['reported_time']<=end_date)]
    incidents_subset = incidents_subset[['season', 'fuel_type', 'reported_time', 'containment_time_hr', 'fire_area_ha', 'latitude', 'longitude', 'point', 'geometry', 'spreading_fire_flags', 'AM60_moisture', 'Curing_%', 'Wind']]
    incidents_subset['reported_date'] = pd.to_datetime(incidents_subset['reported_time'].dt.date)
    
    #further filtering needed:
    #For incidents >200 ha, we need it to contain a mapped geometry.
    incidents_subset = incidents_subset[(((incidents_subset['geometry']!=None) & (incidents_subset['fire_area_ha']>=200)) | (incidents_subset['fire_area_ha']<200))]
    
    #Load shapefile for FWDs:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
    
    #Spatial join to get FWD that the incident is in:
    incidents_subset = geopandas.GeoDataFrame(incidents_subset, geometry='point', crs=shp_in.crs)
    #Before joining - change projection to GDA2020/VicGrids for proper consistency:
    shp_in.to_crs("EPSG:7899", inplace=True)
    incidents_subset.to_crs("EPSG:7899", inplace=True)
    #incidents_subset = geopandas.tools.sjoin(incidents_subset, shp_in, how='left', predicate='within')
    incidents_subset = geopandas.tools.sjoin_nearest(incidents_subset, shp_in, how='left')
    
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

    #Merge incident count, fire area onto moisture data:
    moisture_incident_count = pd.merge(left=df_moisture_curing, right=incidents_count, how='left', left_on=['date','region'], right_on=['reported_date','Area_Name'])
    moisture_incident_count = pd.merge(left=moisture_incident_count, right=incidents_total_ha, how='left', left_on=['date','region'], right_on=['reported_date', 'Area_Name'])
    #If the days don't have an incident count, it merges as nan. Fill those with 0s because there are actually no incidents.
    moisture_incident_count['point'] = moisture_incident_count['point'].fillna(0)
    #Do the same for fire area and containment time. No incidents=no area burnt.
    #TODO: Consider whether we want to drop these instead? Do we want days with no area burnt?
    moisture_incident_count['fire_area_ha'] = moisture_incident_count['fire_area_ha'].fillna(0)
    moisture_incident_count['containment_time_hr'] = moisture_incident_count['containment_time_hr'].fillna(0)
    #Set up binary - if there was an incident, set as 1, otherwise 0. For binomial count.
    moisture_incident_count['incidents_on_day'] = np.where(moisture_incident_count['point']>0, 1,0)
    
    #Clean out data with no moisture (because areas are too small)
    moisture_incident_count = moisture_incident_count[~moisture_incident_count['curing_%'].isna()]
    
    
    #Subset to a specific FWD:
    region_name = 'East Gippsland'
    moisture_incident_count = moisture_incident_count[moisture_incident_count['region']==region_name]
    incidents_subset = incidents_subset[incidents_subset['Area_Name']==region_name]
    
    #Take square root of area, to calculate a "characteristic fire length"
    #Area measured in hectares. So sqrt gives us units of 100m, so divide by 10 (multiply by 0.1) to give km
    #as the length.
    incidents_subset['char_length'] = np.sqrt(incidents_subset['fire_area_ha'])*0.1
    moisture_incident_count['char_length'] = np.sqrt(moisture_incident_count['fire_area_ha'])*0.1

    
    #%%
    #Now time to fit the binomial hurdle step models.
    spline_number = 7

    #Fit GAMs:
    gam_binomial_moisture = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    gam_binomial_curing = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values)
    null_reg_ = LogisticGAM(s(0, n_splines=spline_number, spline_order=3)).fit(np.full(len(moisture_incident_count), 1).reshape(-1,1), moisture_incident_count['incidents_on_day'].values)

    #Calculate pseudo R2 metric for moisture:    
    moist_psr2 = pseudo_rsq(gam_binomial_moisture, null_reg_, moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values)
    print("Pseudo R2 for binomial GAM on moisture: %1.4f " % moist_psr2)
    #Now the same for curing.
    curing_good = pseudo_rsq(gam_binomial_curing,null_reg_, moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values)
    print("Pseudo R2 binomial GAM on curing: %1.4f" % curing_good)
    
    #plot?
    fig, axs = plt.subplots(1)
    xx = gam_binomial_moisture.generate_X_grid(term=0).flatten()  #apparently this creates an array of single value arrays? Messy...
    binomial_moisture_predict = gam_binomial_moisture.predict_mu(xx)
    axs.plot(xx, binomial_moisture_predict, label='GAM')
    axs.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axs.set_title('Predicted fuel moisture')
    axs.set_xlabel('Estimated fuel moisture (%)')
    axs.set_ylabel("fires/no fires")

    figa, axsa = plt.subplots(1)    
    xx2 = gam_binomial_curing.generate_X_grid(term=0).flatten()
    binomial_curing_pred = gam_binomial_curing.predict_mu(xx2)
    axsa.plot(xx2, binomial_curing_pred)
    axsa.scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['incidents_on_day'].values, facecolor='gray', edgecolors='none', s=8)
    axsa.set_title('Curing - All')
    axsa.set_xlabel('Curing')
    axsa.set_ylabel("fires/no fires")
    
    #Normalise curing curve and compare to Cruz and Cheney curves.
    y_curing = gam_binomial_curing.predict_mu(xx2)
    y_curing_norm = (y_curing-np.min(y_curing))/(np.max(y_curing)-np.min(y_curing))
    cheney_curve = 1.12/(1+59.2*np.exp(-0.124*(xx2-50)))
    cruz_curve = 1.036/(1+103.98*np.exp(-0.0996*(xx2-20)))
    
    figb, axsb = plt.subplots(1)
    axsb.plot(xx2, y_curing_norm, label='Normalised GAM')
    axsb.plot(xx2, cheney_curve, label='Cheney', color='orange')
    axsb.plot(xx2, cruz_curve, label='Cruz', color='green')
    axsb.set_title('Curing - Normalised GAM, Cheney, Cruz functions')
    axsb.set_xlabel('Curing')
    axsb.set_ylabel("func")
    axsb.legend()
    
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
       
    
    #OK let's try this way to make a surface plot.
    xx3 = gam_binomial_2d.generate_X_grid(term=0, meshgrid=False)
    yy3 = gam_binomial_2d.generate_X_grid(term=1, meshgrid=False)
    xxx = np.empty([len(xx3), len(yy3)])
    yyy = np.empty([len(xx3), len(yy3)])
    Z = np.empty([len(xx3), len(yy3)])
    
    for i in range(0,len(xx3)):
        xxx[:,i] = xx3[:,0]
        yyy[i,:] = yy3[:,1]
        xx3[:,1] = yy3[i,1]
        Z[:,i] = gam_binomial_2d.predict_mu(xx3)
    
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


    #We know the shapes now. Now need to fit a function to this.
    #FMC: Assume a negative exponential.
    def func_neg_exp(x, a, b):
        return a * np.exp(-b*x)
    moisture_name = []
    moisture_a = []
    moisture_b = []
    
    def func_exp_gen(x, a, b, x0):
        return a*np.exp(b*(x-x0))
    curing_name_exp = []
    curing_a_exp = []
    curing_b_exp = []
    curing_x0_exp = []
    
    moisture_fit_bi = curve_fit(func_neg_exp, xx.flatten(), binomial_moisture_predict)
    moisture_fit_bi_r2 = 1 - (np.sum((binomial_moisture_predict - func_neg_exp(xx, moisture_fit_bi[0][0], moisture_fit_bi[0][1]))**2)
                              /(np.sum((binomial_moisture_predict - np.mean(binomial_moisture_predict))**2)))
    print('R2 of exponential fit to moisture GAM: %1.4f ' % moisture_fit_bi_r2)
    axs.plot(xx.flatten(), func_neg_exp(xx, moisture_fit_bi[0][0], moisture_fit_bi[0][1]), color='r', label='function fit')
    axs.legend()
    moisture_name.append('binomial')
    moisture_a.append(moisture_fit_bi[0][0])
    moisture_b.append(moisture_fit_bi[0][1])
    
    #Curing: Assume a sigmoid shape same as Cheney, Cruz curves.
    def func_sigmoid(x, b, x0):
        return 1/(1+np.exp(-b*(x-x0)))
    
    curing_name = []
    curing_a = []
    curing_b = []
    curing_x0 = []
    
    curing_fit_bi = curve_fit(func_sigmoid, xx2.flatten(), y_curing_norm, p0=[0.1, 95])
    curing_fit_bi_r2 = 1 - (np.sum((y_curing_norm - func_sigmoid(xx2, curing_fit_bi[0][0], curing_fit_bi[0][1]))**2)
                              /(np.sum((y_curing_norm - np.mean(y_curing_norm))**2)))
    print('R2 of exponential fit to curing GAM: %1.4f ' % curing_fit_bi_r2)
    axsb.plot(xx2.flatten(), func_sigmoid(xx2, curing_fit_bi[0][0], curing_fit_bi[0][1]), color='r', label='function fit')
    axsb.legend()
    curing_name.append('binomial')
    curing_a.append(1)
    curing_b.append(curing_fit_bi[0][0])
    curing_x0.append(curing_fit_bi[0][1])
    
    
    
    print("*******************************************")
    
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
#    xx4 = gam_poisson_moisture.generate_X_grid(term=0).flatten()
    xx4 = np.arange(0,22, step=0.2)
    poisson_moisture_predict = gam_poisson_moisture.predict_mu(xx4)
    current_curve = np.exp(-0.108*xx4) * gam_poisson_moisture.predict(0)  #current curve is in grass model now. Then just multiply by maximum at moisture=0 to scale
    axsc.plot(xx4, poisson_moisture_predict, label='GAM')
    axsc.plot(xx4, current_curve, color='k', label='Existing phi_M')
    axsc.scatter(moisture_incident_count['AM60_min'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
    axsc.set_title('Predicted fuel moisture')
    axsc.set_xlabel('Estimated fuel moisture (%)')
    axsc.set_ylabel("Number of fires")
    axsc.set_ylim(0,2)
    
    #Now the same for curing.
    figb, axsb = plt.subplots(1)    
    xx4b = gam_poisson_curing.generate_X_grid(term=0).flatten()
    poisson_curing_pred = gam_poisson_curing.predict_mu(xx4b)
    cheney_curve = 1.12/(1+59.2*np.exp(-0.124*(xx4b-50)))
    cruz_curve = 1.036/(1+103.98*np.exp(-0.0996*(xx4b-20)))
    cheney_curve_p = cheney_curve*np.max(poisson_curing_pred)   #scale Cheney and Cruz curves to the max in the GAM
    cruz_curve_p = cruz_curve*np.max(poisson_curing_pred)    
    axsb.plot(xx4b, poisson_curing_pred, label='GAM', color='tab:blue')
    axsb.plot(xx4b, cheney_curve_p, label="Cheney coeff", color='orange')
    axsb.plot(xx4b, cruz_curve_p, label='Cruz coeff', color='green')
    axsb.scatter(moisture_incident_count['curing_%'].values, moisture_incident_count['point'].values, facecolor='gray', edgecolors='none', s=8)
    axsb.set_title('Curing - All')
    axsb.set_xlabel('Curing')
    axsb.set_ylabel("Number of fires")
    axsb.set_ylim(0,0.2)
    axsb.legend()    
    #2D version:
    gam_poisson_2d = PoissonGAM(s(0, n_splines=spline_number, spline_order=3)
                           +s(1, n_splines=spline_number, spline_order=3, constraints='monotonic_inc')
                           +te(0,1, n_splines=(spline_number,spline_number), constraints=('monotonic_dec', 'monotonic_inc'))).fit(moisture_incident_count[['AM60_min', 'curing_%']].values, moisture_incident_count['point'].values)

    #Goodness of fit for 2D binomial GAM:
    poiss_2d_good = gam_poisson_2d._estimate_r2(X=moisture_incident_count[['AM60_min', 'curing_%']], y=moisture_incident_count['point'])['explained_deviance']
    print("Pseudo R2 of Poisson 2D GAM: %1.4f" % poiss_2d_good)

    
    
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
        Z[:,i] = gam_poisson_2d.predict_mu(xx5)
    
    fig4, axs4 = plt.subplots(1,2, subplot_kw={"projection": "3d"})
    axs4[0].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs4[0].set_title('GAM')
 
    #Fit functional curves and add to plot for comparison:
    moisture_fit_poisson = curve_fit(func_neg_exp, xx4.flatten(), poisson_moisture_predict)
    moisture_fit_poisson_r2 = 1 - (np.sum((poisson_moisture_predict - func_neg_exp(xx4, moisture_fit_poisson[0][0], moisture_fit_poisson[0][1]))**2)
                              /(np.sum((poisson_moisture_predict - np.mean(poisson_moisture_predict))**2)))
    print('R2 of exponential fit to moisture Poisson GAM: %1.4f ' % moisture_fit_poisson_r2)
    axsc.plot(xx4.flatten(), func_neg_exp(xx4, moisture_fit_poisson[0][0], moisture_fit_poisson[0][1]), color='r', label='function fit')
    axsc.legend()
    moisture_name.append('Poisson')
    moisture_a.append(moisture_fit_poisson[0][0])
    moisture_b.append(moisture_fit_poisson[0][0])
    
    #For any curing function not predicting a probability - the maximum isn't necessarily 1.
    #So need this as a function parameter
    def func_sigmoid_scaled(x, a, b, x0):
        return a/(1+np.exp(-b*(x-x0)))
    
    curing_fit_poisson = curve_fit(func_sigmoid_scaled, xx4b, poisson_curing_pred,p0=[0.8, 1, 50], bounds=([0,0,0],[1,1000,100]), maxfev=5000)
    curing_fit_poisson_r2 = 1 - (np.sum((poisson_curing_pred - func_sigmoid_scaled(xx4b, curing_fit_poisson[0][0], curing_fit_poisson[0][1], curing_fit_poisson[0][2]))**2)
                              /(np.sum((poisson_curing_pred - np.mean(poisson_curing_pred))**2)))
    print('R2 of sigmoid fit to curing Poisson GAM: %1.4f ' % curing_fit_poisson_r2)
    axsb.plot(xx4b, func_sigmoid_scaled(xx4b, curing_fit_poisson[0][0], curing_fit_poisson[0][1], curing_fit_poisson[0][2]), color='r', label='function fit')
    axsb.legend()
    curing_name.append('Poisson')
    curing_a.append(curing_fit_poisson[0][0])
    curing_b.append(curing_fit_poisson[0][1])
    curing_x0.append(curing_fit_poisson[0][2])

    #Finally, let's try the 2D GAM. First define a function that is the product of moisture and curing curves.
    #This product exists in the existing grassland model.    
    def func_2d(X, a0, b0, a1,b1, y0):
        x = X[:,0]
        y = X[:,1]
        return (a0*np.exp(-b0*x))*(a1/(1+np.exp(-b1*(y-y0))))
    twod_a0 = []
    twod_b0 = []
    twod_a1 = []
    twod_b1 = []
    twod_y0 = []
    
    #Set up x (inputs) and y (target), then curve fit
    x_data_ = np.c_[xxx.flatten(), yyy.flatten()]
    y_data_ = gam_poisson_2d.predict_mu(x_data_)
    fit_poisson_2d = curve_fit(func_2d, xdata=x_data_,ydata=y_data_, p0=[0.5, 0.1, 1,1,80], maxfev=5000)
    fit_prms = fit_poisson_2d[0]
    fit_poisson_2d_r2 = 1 - (
                            np.sum((y_data_ - func_2d(x_data_, fit_prms[0], fit_prms[1], fit_prms[2],fit_prms[3],fit_prms[4]))**2)
                             /(np.sum((y_data_ - np.mean(y_data_))**2))
                             )
    print('R2 of 2D fit to curing Poisson GAM: %1.4f' % fit_poisson_2d_r2)
    
    #Kind of have to make this plot the hard way.
    for i in range(0, len(xx5)):
        Z[:,i] = func_2d(np.c_[xxx[:,i],yyy[:,i]], fit_prms[0], fit_prms[1], fit_prms[2], fit_prms[3], fit_prms[4])
    
    axs4[1].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs4[1].set_title('function fit')
    fig4.suptitle('2D Poisson GAM - number of incidents/region/day')   
    twod_a0.append(fit_prms[0])
    twod_b0.append(fit_prms[1])
    twod_a1.append(fit_prms[2])
    twod_b1.append(fit_prms[3])
    twod_y0.append(fit_prms[4])

    print("*******************************************")
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
    
    fig6, axs6 = plt.subplots(1,2, figsize=(11,5))
    xx6 = gam_area_fmc.generate_X_grid(term=0).flatten()
    xx6_cur = gam_area_cur.generate_X_grid(term=0).flatten()
    area_moist_pred = gam_area_fmc95.predict(xx6)
    area_cur_pred = gam_area_cur95.predict(xx6_cur)
    moisture_current_curve = np.exp(-0.108*xx6) * gam_area_fmc95.predict(0)
    axs6[0].scatter(moisture_incident.values, fire_area_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs6[0].plot(xx6, gam_area_fmc.predict(xx6), color='k')
    axs6[0].plot(xx6, area_moist_pred, color='tab:blue')
    axs6[0].plot(xx6, moisture_current_curve, color='green')
#    axs6[0].plot(xx8, gam_area_fmc95_ci90.predict(xx8), color='orange')
#    axs6[0].plot(xx8, gam_area_fmc95_ci10.predict(xx8), color='orange')
    axs6[0].set_ylabel('Total area burnt (ha)')
    axs6[0].set_ylim(0,8000)
    axs6[0].set_xlabel('Moisture (McArthur) %')
    axs6[0].set_title('Fuel moisture', fontsize=16)
    axs6[1].scatter(curing_incident.values, fire_area_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs6[1].plot(xx6_cur, gam_area_cur.predict(xx6_cur), color='k')
    axs6[1].plot(xx6_cur, area_cur_pred, color='tab:blue')    
    cheney_curve_area = cheney_curve*gam_area_cur95.predict(100)
    cruz_curve_area = cruz_curve*gam_area_cur95.predict(100)
    axs6[1].plot(xx6_cur, cheney_curve_area, color='orange')
    axs6[1].plot(xx6_cur, cruz_curve_area, color='green')
    axs6[1].set_ylim(0,1500)
    axs6[1].set_xlabel('Curing %')
    axs6[1].set_title('Curing', fontsize=16)
#    axs6[1].legend(['points','mean', '95%'])
    fig6.suptitle('Total Area burnt, by incident', fontsize=20)
    
    
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

    fig7, axs7 = plt.subplots(1,2, figsize=(11,6), subplot_kw={"projection": "3d"})
    axs7[0].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs7[0].set_title('Area prediction GAM')
    
    
    moisture_fit_area = curve_fit(func_neg_exp, xx6, area_moist_pred)
    moisture_fit_area_r2 = 1 - (np.sum((area_moist_pred - func_neg_exp(xx6, moisture_fit_area[0][0], moisture_fit_area[0][1]))**2)
                              /(np.sum((area_moist_pred - np.mean(area_moist_pred))**2)))
    print('R2 of exponential fit to moisture area GAM: %1.4f ' % moisture_fit_area_r2)
    axs6[0].plot(xx6.flatten(), func_neg_exp(xx6, moisture_fit_area[0][0], moisture_fit_area[0][1]), color='r', label='function fit')
    axs6[0].legend(['points','mean', str(int(expectile_set*100))+'%','existing phi_M', 'function fit'])
    moisture_name.append('Area_incident')
    moisture_a.append(moisture_fit_area[0][0])
    moisture_b.append(moisture_fit_area[0][1])
    

    curing_fit_area = curve_fit(func_sigmoid_scaled, xx6_cur.flatten(), gam_area_cur95.predict(xx6_cur.flatten()), p0=[1200, 0.02, 105], bounds=([0,0,0],[20000,1000,500]), maxfev=5000 )
    curing_fit_area_r2 = 1 - (np.sum((area_cur_pred - func_sigmoid_scaled(xx6_cur, curing_fit_area[0][0], curing_fit_area[0][1], curing_fit_area[0][2]))**2)
                              /(np.sum((area_cur_pred - np.mean(area_cur_pred))**2)))
    print('R2 of sigmoid fit to curing area GAM: %1.4f ' % curing_fit_area_r2)
    axs6[1].plot(xx6_cur.flatten(), func_sigmoid_scaled(xx6_cur.flatten(), curing_fit_area[0][0],  curing_fit_area[0][1], curing_fit_area[0][2]), color='r', label='function fit')
    axs6[1].legend(['points','mean', str(int(expectile_set*100))+'%','Cheney','Cruz','function_fit'])
    curing_a.append(curing_fit_area[0][0])
    curing_b.append(curing_fit_area[0][1])
    curing_x0.append(curing_fit_area[0][2])
    
    #2D fit and plot:
    x_data_ = np.c_[xxx.flatten(), yyy.flatten()]
    y_data_ = gam_area_2d95.predict(x_data_)
    fit_area_2d = curve_fit(func_2d, xdata=x_data_,ydata=y_data_, p0=[1000, 0.1, 100,0.5,70], maxfev=7000)
    fit_prms = fit_area_2d[0]
    fit_area_2d_r2 = 1 - (
                            np.sum((y_data_ - func_2d(x_data_, fit_prms[0], fit_prms[1], fit_prms[2],fit_prms[3],fit_prms[4]))**2)
                             /(np.sum((y_data_ - np.mean(y_data_))**2))
                             )
    print('R2 of fit to 2D area GAM: %1.4f' % fit_area_2d_r2)
    
    #Kind of have to make this plot the hard way.
    for i in range(0, len(xx9)):
        Z[:,i] = func_2d(np.c_[xxx[:,i],yyy[:,i]], fit_prms[0], fit_prms[1], fit_prms[2], fit_prms[3], fit_prms[4])
    
    axs7[1].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs7[1].set_title('function fit')
    fig7.suptitle('Area of Incident vs curing/fuel moisture')   
    
    twod_a0.append(fit_prms[0])
    twod_b0.append(fit_prms[1])
    twod_a1.append(fit_prms[2])
    twod_b1.append(fit_prms[3])
    twod_y0.append(fit_prms[4])
    
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
    
    #Calculate our goodness of fit measures:
    len_moist_res95 = length_incident.values - gam_len_fmc95.predict(moisture_incident.values)
    len_cur_res95 = length_incident.values - gam_len_cur95.predict(curing_incident.values)
    len_2d_res95 = length_incident.values - gam_len_2d95.predict(incidents_subset[['AM60_moisture', 'Curing_%']].values)
    len_null95 = length_incident.values - expectile(length_incident.values, alpha=expectile_set)
    
    
    len_moist_good = goodfit(len_moist_res95, len_null95, expectile_set)
    len_cur_good = goodfit(len_cur_res95, len_null95, expectile_set)
    len_2d_good = goodfit(len_2d_res95, len_null95, expectile_set)
    print('Goodness of fit of characteristic length GAM on moisture only: %1.3f' % len_moist_good)
    print('Goodness of fit of characteristic length GAM on curing only: %1.3f' % len_cur_good)
    print('Goodness of fit of 2D characteristic length GAM: %1.3f' % len_2d_good)
    
    cruz_curing1d = 1.036/(1+103.98*np.exp(-0.0996*(xx6_cur-20)))*gam_len_cur95.predict(100)
    cheney_curing1d = 1.12/(1+59.2*np.exp(-0.124*(xx6_cur-50)))*gam_len_cur95.predict(100)
    
    fig10, axs10 = plt.subplots(1,2, figsize=(11,5))
    len_moist_pred = gam_len_fmc95.predict(xx6)
    len_cur_pred = gam_len_cur95.predict(xx6_cur)
    axs10[0].scatter(moisture_incident.values, length_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs10[0].plot(xx6, len_moist_pred, color='tab:blue')    
    axs10[0].set_ylabel('Length (km)')
    axs10[0].set_ylim(0,5)
    axs10[0].set_xlabel('Moisture (McArthur) %')
    axs10[0].set_title('Fuel moisture', fontsize=16)
    axs10[0].legend(['points',str(int(expectile_set*100))+'%GAM'])
    axs10[1].scatter(curing_incident.values, length_incident.values, facecolor='gray', edgecolors='none', s=8)
    axs10[1].plot(xx6_cur, len_cur_pred, color='tab:blue')    
    axs10[1].plot(xx6_cur, cruz_curing1d, color='green')
    axs10[1].plot(xx6_cur, cheney_curing1d, color='orange')
    axs10[1].set_ylim(0,4)
    axs10[1].set_xlabel('Curing %')
    axs10[1].set_title('Curing', fontsize=16)
    axs10[1].legend(['points','95%GAM', 'Cruz', 'Cheney'])
    fig10.suptitle('Characteristic fire length, by incident', fontsize=20)


    res_cheney1d = length_incident - (1.12/(1+59.2*np.exp(-0.124*(curing_incident-50))))*gam_len_cur95.predict(100)
    good_cheney1d = goodfit(res_cheney1d, len_null95, expectile_set)
    print(good_cheney1d)
    xx10 = gam_len_2d95.generate_X_grid(term=0, meshgrid=False)
    yy10 = gam_len_2d95.generate_X_grid(term=1, meshgrid=False)
        
    xxx = np.empty([len(xx10), len(yy10)])
    yyy = np.empty([len(xx10), len(yy10)])
    Z = np.empty([len(xx10), len(yy10)])
    for i in range(0,len(xx10)):
            xxx[:,i] = xx10[:,0]
            yyy[i,:] = yy10[:,1]
            xx10[:,1] = yy10[i,1]
            Z[:,i] = gam_len_2d95.predict(xx10)

    fig11, axs11 = plt.subplots(1,2, figsize=(11,6), subplot_kw={"projection": "3d"})
    axs11[0].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs11[0].set_title('Char. length vs curing and fuel moisture')
                
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
    
    
    moisture_fit_len = curve_fit(func_neg_exp, xx6.flatten(), gam_len_fmc95.predict(xx6))
    moisture_fit_len_r2 = 1 - (np.sum((len_moist_pred - func_neg_exp(xx6, moisture_fit_len[0][0], moisture_fit_len[0][1]))**2)
                              /(np.sum((len_moist_pred - np.mean(len_moist_pred))**2)))
    print('R2 of exponential fit to moisture length GAM: %1.4f ' % moisture_fit_len_r2)
    axs10[0].plot(xx6.flatten(), func_neg_exp(xx6, moisture_fit_len[0][0], moisture_fit_len[0][1]), color='r', label='function fit')
    axs10[0].legend(['points','95%', 'function fit'])
    moisture_name.append('Length_incident')
    moisture_a.append(moisture_fit_len[0][0])
    moisture_b.append(moisture_fit_len[0][1])
    
    curing_fit_len = curve_fit(func_sigmoid_scaled, xx6_cur.flatten(), gam_len_cur95.predict(xx6_cur), p0=[2.0, 5, 60], bounds=([1,0,0], [10,10,1000]), maxfev=5000)
    curing_fit_len_r2 = 1 - (np.sum((len_cur_pred - func_sigmoid_scaled(xx6_cur, curing_fit_len[0][0], curing_fit_len[0][1], curing_fit_len[0][2]))**2)
                              /(np.sum((len_cur_pred - np.mean(len_cur_pred))**2)))
    print('R2 of sigmoid fit to curing length GAM: %1.4f ' % curing_fit_len_r2)
    axs10[1].plot(xx6_cur.flatten(), func_sigmoid_scaled(xx6_cur.flatten(), curing_fit_len[0][0],  curing_fit_len[0][1], curing_fit_len[0][2]), color='r', label='function fit')
    axs10[1].legend(['points',str(int(expectile_set*100))+'%', 'Cruz', 'Cheney', 'function fit'])
    curing_a.append(curing_fit_len[0][0])
    curing_b.append(curing_fit_len[0][1])
    curing_x0.append(curing_fit_len[0][2])
    
    #2D fit and plot:
    x_data_ = np.c_[xxx.flatten(), yyy.flatten()]
    y_data_ = gam_len_2d95.predict(x_data_)
    fit_len_2d = curve_fit(func_2d, xdata=x_data_,ydata=y_data_, p0=[2, 0.1,2,0.11,50], maxfev=5000)
    fit_prms = fit_len_2d[0]
    fit_len_2d_r2 = 1 - (
                            np.sum((y_data_ - func_2d(x_data_, fit_prms[0], fit_prms[1], fit_prms[2],fit_prms[3],fit_prms[4]))**2)
                             /(np.sum((y_data_ - np.mean(y_data_))**2))
                             )
    print('R2 of fit to 2D length GAM: %1.4f' % fit_len_2d_r2)
    
    #Kind of have to make this plot the hard way.
    for i in range(0, len(xx10)):
        Z[:,i] = func_2d(np.c_[xxx[:,i],yyy[:,i]], fit_prms[0], fit_prms[1], fit_prms[2], fit_prms[3], fit_prms[4])
    
    axs11[1].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs11[1].set_title('function fit')
    fig11.suptitle('Length of Incident vs curing/fuel moisture')  
    
    twod_a0.append(fit_prms[0])
    twod_b0.append(fit_prms[1])
    twod_a1.append(fit_prms[2])
    twod_b1.append(fit_prms[3])
    twod_y0.append(fit_prms[4])
    
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


    #Calculate our goodness of fit measures:
    len_moist_res95 = length_region.values - gam_len_fmc95.predict(moisture_region.values)
    len_cur_res95 = length_region.values - gam_len_cur95.predict(curing_region.values)
    len_2d_res95 = length_region.values - gam_len_2d95.predict(moisture_incident_count[['AM60_min', 'curing_%']].values)
    len_null95 = length_region.values - expectile(length_region.values, alpha=expectile_set)
    
    
    len_moist_good = goodfit(len_moist_res95, len_null95, expectile_set)
    len_cur_good = goodfit(len_cur_res95, len_null95, expectile_set)
    len_2d_good = goodfit(len_2d_res95, len_null95, expectile_set)
    print("***By district:***")
    print('Goodness of fit of characteristic length GAM on moisture only: %1.3f' % len_moist_good)
    print('Goodness of fit of characteristic length GAM on curing only: %1.3f' % len_cur_good)
    print('Goodness of fit of 2D characteristic length GAM: %1.3f' % len_2d_good)
    
    xx7 = gam_len_fmc95.generate_X_grid(term=0).flatten()
    xx7_cur = gam_len_cur95.generate_X_grid(term=0).flatten()
    cruz_curing1d = 1.036/(1+103.98*np.exp(-0.0996*(xx7_cur-20)))*gam_len_cur95.predict(100)
    cheney_curing1d = 1.12/(1+59.2*np.exp(-0.124*(xx7_cur-50)))*gam_len_cur95.predict(100)
    
    fig12, axs12 = plt.subplots(1,2, figsize=(11,5))
    len_moist_pred = gam_len_fmc95.predict(xx7)
    len_cur_pred = gam_len_cur95.predict(xx7_cur)
    axs12[0].scatter(moisture_region.values, length_region.values, facecolor='gray', edgecolors='none', s=8)
    axs12[0].plot(xx7, len_moist_pred, color='tab:blue')    
    axs12[0].set_ylabel('Length (km)')
    axs12[0].set_ylim(0,2)
    axs12[0].set_xlabel('Moisture (McArthur) %')
    axs12[0].set_title('Fuel moisture', fontsize=16)
    axs12[0].legend(['points',str(int(expectile_set*100))+'%GAM'])
    axs12[1].scatter(curing_region.values, length_region.values, facecolor='gray', edgecolors='none', s=8)
    axs12[1].plot(xx7_cur, len_cur_pred, color='tab:blue')    
    axs12[1].plot(xx7_cur, cruz_curing1d, color='green')
    axs12[1].plot(xx7_cur, cheney_curing1d, color='orange')
    axs12[1].set_ylim(0,2)
    axs12[1].set_xlabel('Curing %')
    axs12[1].set_title('Curing', fontsize=16)
    axs12[1].legend(['points','95%GAM', 'Cruz', 'Cheney'])
    fig12.suptitle('Characteristic fire length, by district total', fontsize=20)

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

    fig14, axs14 = plt.subplots(1,2, figsize=(11,6), subplot_kw={"projection": "3d"})
    axs14[0].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs14[0].set_title('GAM')
            
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

    
    moisture_fit_len_reg = curve_fit(func_neg_exp, xx7.flatten(), gam_len_fmc95.predict(xx7))
    moisture_fit_len_r2 = 1 - (np.sum((len_moist_pred - func_neg_exp(xx7, moisture_fit_len_reg[0][0], moisture_fit_len_reg[0][1]))**2)
                              /(np.sum((len_moist_pred - np.mean(len_moist_pred))**2)))
    print('R2 of exponential fit to moisture length GAM (region total): %1.4f ' % moisture_fit_len_r2)
    axs12[0].plot(xx7.flatten(), func_neg_exp(xx7, moisture_fit_len_reg[0][0], moisture_fit_len_reg[0][1]), color='r', label='function fit')
    axs12[0].legend(['points',str(int(expectile_set*100))+'%', 'function fit'])
    moisture_name.append('Length_region')
    moisture_a.append(moisture_fit_len_reg[0][0])
    moisture_b.append(moisture_fit_len_reg[0][1])
    
    curing_fit_len_reg = curve_fit(func_sigmoid_scaled, xx7_cur.flatten(), gam_len_cur95.predict(xx7_cur), p0=[1.0, 5, 60], bounds=([0,0,0], [4,10,200]), maxfev=5000)
    curing_fit_len_r2 = 1 - (np.sum((len_cur_pred - func_sigmoid_scaled(xx7_cur, curing_fit_len_reg[0][0], curing_fit_len_reg[0][1], curing_fit_len_reg[0][2]))**2)
                              /(np.sum((len_cur_pred - np.mean(len_cur_pred))**2)))
    print('R2 of sigmoid fit to curing length GAM (region total): %1.4f ' % curing_fit_len_r2)
    axs12[1].plot(xx7_cur.flatten(), func_sigmoid_scaled(xx7_cur.flatten(), curing_fit_len_reg[0][0],  curing_fit_len_reg[0][1], curing_fit_len_reg[0][2]), color='r', label='function fit')
    axs12[1].legend(['points',str(int(expectile_set*100))+'%', 'Cruz', 'Cheney', 'function fit'])
    curing_a.append(curing_fit_len_reg[0][0])
    curing_b.append(curing_fit_len_reg[0][1])
    curing_x0.append(curing_fit_len_reg[0][2])
    
    #2D fit and plot:
    x_data_ = np.c_[xxx.flatten(), yyy.flatten()]
    y_data_ = gam_len_2d95.predict(x_data_)
    fit_len_2d = curve_fit(func_2d, xdata=x_data_,ydata=y_data_, p0=[5, 0.1, 1,1,50], maxfev=7000)
    fit_prms = fit_len_2d[0]
    fit_len_2d_r2 = 1 - (
                            np.sum((y_data_ - func_2d(x_data_, fit_prms[0], fit_prms[1], fit_prms[2],fit_prms[3],fit_prms[4]))**2)
                             /(np.sum((y_data_ - np.mean(y_data_))**2))
                             )
    print('R2 of fit to 2D length GAM: %1.4f' % fit_len_2d_r2)
    
    #Kind of have to make this plot the hard way.
    for i in range(0, len(xx10)):
        Z[:,i] = func_2d(np.c_[xxx[:,i],yyy[:,i]], fit_prms[0], fit_prms[1], fit_prms[2], fit_prms[3], fit_prms[4])
    
    axs14[1].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs14[1].set_title('function fit')
    fig14.suptitle('Length total in region vs curing/fuel moisture')  
    
    twod_a0.append(fit_prms[0])
    twod_b0.append(fit_prms[1])
    twod_a1.append(fit_prms[2])
    twod_b1.append(fit_prms[3])
    twod_y0.append(fit_prms[4])
    
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
    
    gam_area_fmc95 = ExpectileGAM(s(0,n_splines=spline_number, spline_order=3,), expectile=expectile_set).fit(moisture_region.values, fire_area_region.values)
    
    
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
    xx11 = np.arange(0,20,0.2)
    xx11_cur = np.arange(0,100,0.5)
    """
    xx11 = gam_area_fmc95.generate_X_grid(term=0)
    xx11_cur = gam_area_cur95.generate_X_grid(term=0)
    """
    area_moist_pred = gam_area_fmc95.predict(xx11)
    area_cur_pred = gam_area_cur95.predict(xx11_cur)
    axs16[0].scatter(moisture_region.values, fire_area_region.values, facecolor='gray', edgecolors='none', s=8)
    axs16[0].plot(xx11, gam_area_fmc95.predict(xx11), color='tab:blue')    
    axs16[0].set_ylabel('Total area burnt (ha)')
    axs16[0].set_ylim(0,2000)
    axs16[0].set_xlabel('Moisture (McArthur) %')
    axs16[0].set_title('Fuel moisture', fontsize=16)
    axs16[0].legend(['points','mean', str(int(expectile_set*100))+'%'])
    axs16[1].scatter(curing_region.values, fire_area_region.values, facecolor='gray', edgecolors='none', s=8)
    axs16[1].plot(xx11_cur, gam_area_cur95.predict(xx11_cur), color='tab:blue')
    cruz_curve_area_r = cruz_curve*gam_area_cur95.predict(100)
    cheney_curve_area_r = cheney_curve*gam_area_cur95.predict(100)
    axs16[1].plot(xx6_cur, cruz_curve_area_r, color='green')
    axs16[1].plot(xx6_cur, cheney_curve_area_r, color='orange')
    axs16[1].set_ylim(0,500)
    axs16[1].set_xlabel('Curing %')
    axs16[1].set_title('Curing', fontsize=16)
    fig16.suptitle('Total Area burnt, by region', fontsize=20)
    
  
    
    
    xx16 = gam_area_2d95.generate_X_grid(term=0, meshgrid=False)
    yy16 = gam_area_2d95.generate_X_grid(term=1, meshgrid=False)
    
    xxx = np.empty([len(xx16), len(yy16)])
    yyy = np.empty([len(xx16), len(yy16)])
    Z = np.empty([len(xx16), len(yy16)])
    for i in range(0,len(xx16)):
        xxx[:,i] = xx16[:,0]
        yyy[i,:] = yy16[:,1]
        xx16[:,1] = yy16[i,1]
        Z[:,i] = gam_area_2d95.predict(xx16)

    fig17, axs17 = plt.subplots(1,2, figsize=(11,6), subplot_kw={"projection": "3d"})
    axs17[0].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs17[0].set_title('Area prediction vs curing and fuel moisture')

    
    moisture_fit_area_reg = curve_fit(func_neg_exp, xx11.flatten(), gam_area_fmc95.predict(xx11))
    moisture_fit_area_r2 = 1 - (np.sum((area_moist_pred - func_neg_exp(xx11, moisture_fit_area_reg[0][0], moisture_fit_area_reg[0][1]))**2)
                              /(np.sum((area_moist_pred - np.mean(area_moist_pred))**2)))
    print('R2 of exponential fit to moisture area GAM (region total): %1.4f ' % moisture_fit_area_r2)
    axs16[0].plot(xx11.flatten(), func_neg_exp(xx11, moisture_fit_area_reg[0][0], moisture_fit_area_reg[0][1]), color='r', label='function fit')
    axs16[0].legend(['points','95%', 'function fit'])
    moisture_name.append('Area_region')
    moisture_a.append(moisture_fit_area_reg[0][0])
    moisture_b.append(moisture_fit_area_reg[0][1])

    
#    curing_fit_area_reg = curve_fit(func_sigmoid, xx11_cur.flatten(), gam_area_cur95.predict(xx11_cur.flatten()), p0=[0.02, 95], bounds=([0,80],[0.1,200]), maxfev=5000, )
    curing_fit_area_reg = curve_fit(func_sigmoid_scaled, xx11_cur.flatten(), gam_area_cur95.predict(xx11_cur.flatten()), p0=[1, 0.02, 135], maxfev=5000 )
#    curing_fit_area = curve_fit(func_sigmoid, xx6_cur.flatten(), cheney_curve.flatten())
    curing_fit_area_r2 = 1 - (np.sum((area_cur_pred - func_sigmoid_scaled(xx11_cur, curing_fit_area_reg[0][0], curing_fit_area_reg[0][1], curing_fit_area_reg[0][2]))**2)
                              /(np.sum((area_cur_pred - np.mean(area_cur_pred))**2)))
    print('R2 of sigmoid fit to curing area GAM (region total): %1.4f ' % curing_fit_area_r2)
#    axs16a.plot(xx11_cur.flatten(), func_sigmoid(xx11_cur.flatten(), curing_fit_area_reg[0][0],  curing_fit_area_reg[0][1]), color='r', label='function fit')
    axs16[1].plot(xx11_cur.flatten(), func_sigmoid_scaled(xx11_cur.flatten(), curing_fit_area_reg[0][0],  curing_fit_area_reg[0][1], curing_fit_area_reg[0][2]), color='r', label='function fit')
    axs16[1].legend(['points', str(int(expectile_set*100))+'%', 'Cruz','Cheney', 'function fit'])
    curing_a.append(curing_fit_area_reg[0][0])
    curing_b.append(curing_fit_area_reg[0][1])
    curing_x0.append(curing_fit_area_reg[0][2])



    #2D fit and plot:
    x_data_ = np.c_[xxx.flatten(), yyy.flatten()]
    y_data_ = gam_area_2d95.predict(x_data_)
    fit_area_2d = curve_fit(func_2d, xdata=x_data_,ydata=y_data_, p0=[5, 0.1, 1,1,50], maxfev = 7000)
    fit_prms = fit_area_2d[0]
    fit_area_2d_r2 = 1 - (
        np.sum((y_data_ - func_2d(x_data_, fit_prms[0], fit_prms[1], fit_prms[2],fit_prms[3],fit_prms[4]))**2)
        /(np.sum((y_data_ - np.mean(y_data_))**2))
        )
    print('R2 of fit to 2D length GAM: %1.4f' % fit_area_2d_r2)
        
    #Kind of have to make this plot the hard way.
    for i in range(0, len(xx10)):
            Z[:,i] = func_2d(np.c_[xxx[:,i],yyy[:,i]], fit_prms[0], fit_prms[1], fit_prms[2], fit_prms[3], fit_prms[4])
        
    axs17[1].plot_surface(xxx, yyy, Z, cmap='viridis')
    axs17[1].set_title('function fit')
    fig17.suptitle('Area total in region vs curing/fuel moisture')      
    
    twod_a0.append(fit_prms[0])
    twod_b0.append(fit_prms[1])
    twod_a1.append(fit_prms[2])
    twod_b1.append(fit_prms[3])
    twod_y0.append(fit_prms[4])
    
    print("************************************")
    
    #%%
    
    moisture_out = pd.DataFrame({'Data_fit': moisture_name, 'a': moisture_a, 'b': moisture_b})
    moisture_out.to_csv('C://Users/clark/analysis1/incidents_fmc_data/function_parameters/'+region_name+'/neg_exp_moisture_'+str(int(expectile_set*100))+'.csv')
#    moisture_out.to_csv('C://Users/clark/analysis1/incidents_fmc_data/function_parameters//neg_exp_moisture_'+str(int(expectile_set*100))+'.csv')
    curing_out = pd.DataFrame({'Data_fit': moisture_name, 'a': curing_a, 'b': curing_b, 'x0': curing_x0})
    curing_out.to_csv('C://Users/clark/analysis1/incidents_fmc_data/function_parameters/'+region_name+'/sigmoid_curing_'+str(int(expectile_set*100))+'.csv')
#   curing_out.to_csv('C://Users/clark/analysis1/incidents_fmc_data/function_parameters//sigmoid_curing_'+str(int(expectile_set*100))+'.csv')
    
    twod_out = pd.DataFrame({'Data_fit': moisture_name[1:], 'a0': twod_a0, 'b0': twod_b0, 'a1': twod_a1, 'b1': twod_b1, 'y0': twod_y0})
    twod_out.to_csv('C://Users/clark/analysis1/incidents_fmc_data/function_parameters/'+region_name+'/curingplusmoisture_2d_'+str(int(expectile_set*100))+'.csv')
#    twod_out.to_csv('C://Users/clark/analysis1/incidents_fmc_data/function_parameters/curingplusmoisture_2d_'+str(int(expectile_set*100))+'.csv')
