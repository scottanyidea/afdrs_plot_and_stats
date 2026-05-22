#This script generates outputs necessary for a brief
#case study of AFDRS predicted vs observed conditions, rates of spread, etc.
#This effectively draws on multiple other scripts to put all plots, outputs
#in the one place.

import sys, os
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import geopandas
from shapely.geometry import mapping, Point
import matplotlib.pyplot as plt
import matplotlib.dates as md

sys.path.append(os.path.abspath('afdrs_fbi_recalc'))
sys.path.append(os.path.abspath('afdrs_fbi_recalc/scripts'))
from compile_archived_observations import make_observation_table_from_archive
from recreate_fbi_nc import recreate_fbi_nc_func
from fbi_vic_plot_functions import plot_fbi_and_fdi, plot_fbi_ffdi_gfdi_ratings, plot_varpanel
from plot_recalc_fbi_firearea import plot_fbi_points_in_fire
from calculate_fbi_from_aws import calculate_fbi_for_fueltype, calculate_ffdi_for_awsdata, calculate_gfdi_for_awsdata

year_sel_ = 2026
mon_sel = 1
day_sel = 5
datetime_sel = datetime(year_sel_, mon_sel, day_sel)


#Set context: What was the FBI and FDI across the state (or chosen region) on the day?

##############################################################
#Recalculate FBI and FDI so we have all the data in the one file:
##############################################################

date_str_ = "20260104"    #note: we want the day prior forecast usually
CWD = Path().absolute()
#Set up paths for recalc:
input_grid_folder_pth = Path(r'M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/')
outpath = Path("C://Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data")
#Paths to fuel inputs - ENSURE THESE ARE UP TO DATE!!!
fuel_type_tif_path =  CWD /'afdrs_fbi_recalc' /'data' / 'fuel' / 'fuel-type-authorised-vic-20250818020026.tif'
fuel_type_model_lut_csv_path = CWD /'afdrs_fbi_recalc' / 'data' / 'fuel' / 'fuel-type-model-authorised-vic-20250225011044.csv'
grass_fuel_load_tif_path = CWD / 'afdrs_fbi_recalc' / 'data' / 'fuel' / 'grass-fuel-load-authorised-vic-20251230223702.tif'
time_since_fire_file_name = "C:/Users/clark/analysis1/afdrs_fbi_recalc/data/time_since_fire/20251015T182119_IDZ10162_AUS_FSE_time_since_fire_SFC.nc"
"""
print('*********************************')
print('Recalculating forecast grids')
recreate_fbi_nc_func(date_str_, input_grid_folder_pth, outpath, fuel_type_tif_path, fuel_type_model_lut_csv_path, 
                     time_since_fire_file_name,
                     input_grid_gz=True,
                     grass_fuel_load_default=False, 
                     grass_fuel_load_path=grass_fuel_load_tif_path,
                     truncate_time_index=85,
                     calculate_mcarthur=True, truncated_outputs=False)
"""

#%%
##############################################
#Plot FBI and FDI on the day. And get the values for each region.
##############################################
recalc_file_in = xr.open_dataset(outpath / str("VIC_"+date_str_+"_recalc.nc"))
#recalc_file_in = xr.open_dataset('C://Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_control/VIC_'+date_str_+'_recalc.nc')
forecast_day = 1
#extent = [140.8,145.7,-39,-33.8] #western half Vic
extent = [140.8,150.1,-39.3,-33.8] #All Vic
#extent = [143,146.2,-37.8,-36.3] #Central (well, north central)
#extent = [140.8,144.7,-38.9,-36.7]   #South West
#extent = [145.5, 148.5, -37.7, -35.7]  #North East
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")

print('******************************')
print('Calculating maxima and plotting')

#set strings here - bit of a mess but helps later!
#Note - we add nothing if we want day+1, bc UTC puts us to next day
mon_sel_str_fc = datetime.strftime((datetime.strptime(date_str_, "%Y%m%d")+timedelta(days=forecast_day-1)), "%m")
day_sel_str_fc = datetime.strftime((datetime.strptime(date_str_, "%Y%m%d")+timedelta(days=forecast_day-1)), "%d")
day_sel_str_fcplus1 = datetime.strftime((datetime.strptime(date_str_, "%Y%m%d")+timedelta(days=forecast_day)), "%d")
mon_sel_str_fcplus1 = datetime.strftime((datetime.strptime(date_str_, "%Y%m%d")+timedelta(days=forecast_day)), "%m")
"""
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T12:00:00')
"""
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fc+'-'+day_sel_str_fc+'T21:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str_fcplus1+'-'+day_sel_str_fcplus1+'T21:00:00')

if forecast_day==0:
    start_ind=1
else:
    start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
max_recalc_fdi = recalc_file_in['FDI_SFC'][:,:,start_ind:end_ind+1].max(dim='time', keep_attrs=True)
max_recalc_gfdi = recalc_file_in['GFDI_SFC'][start_ind:end_ind+1,:,:].max(dim='time', keep_attrs=True)
max_recalc_ffdi = recalc_file_in['FFDI_SFC'][start_ind:end_ind+1,:,:].max(dim='time', keep_attrs=True)
max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind+1,:,:].max(dim='time', keep_attrs=True)

plot_fbi_and_fdi(max_recalc_fbi, max_recalc_fdi, shp_in,extent, save_plot=None)

print('******************************')
print('Plotting weather vars')
max_temp = recalc_file_in['T_SFC'][start_ind:end_ind+1,:,:].max(dim='time', keep_attrs=True)
min_rh = recalc_file_in['RH_SFC'][start_ind:end_ind+1,:,:].min(dim='time', keep_attrs=True)
#Get index of max FBI:
arg_maxfbi = recalc_file_in['index_1'][start_ind:end_ind+1,:,:].fillna(-99.).argmax(dim='time', keep_attrs=True) #the fillna forces it to work
median_hr = xr.where(arg_maxfbi>0, arg_maxfbi, np.nan).median().values.item()
arg_maxfbi = xr.where(arg_maxfbi>0, arg_maxfbi, int(median_hr))
wind = recalc_file_in['WindMagKmh_SFC'][start_ind:end_ind+1,:,:][arg_maxfbi]   #get 24h of data in window we want, then address using arg_maxfbi
df = recalc_file_in['DF_SFC'][start_ind:end_ind+1,:,:].max(dim='time', keep_attrs=True)

#shp_in_cut = shp_in[shp_in['Area_Name']=='Glenelg']
plot_varpanel(max_temp, min_rh, wind, df, shp_in, extent, save_plot=None)
#%%
#####################################
#Get FBI values and fuel types at the fire area.
#####################################
"""
print('******************************')
print('Plotting FBI and FTs at fireground')
fuel_types = recalc_file_in['fuel_type'][12,:,:]

#Load fire shapefile, and get specific incident.
#Probably need to match ID in ArcGIS Pro or similar to make sure we get the right one.
fire_shp_in = geopandas.read_file("C:/Users/clark/analysis1/Case_studies/2026_01_09/Obs_area_all_asof_20260116/Obs_area.shp")
#fire_shp_in = geopandas.read_file("C:/Users/clark/analysis1/Case_studies/2026_01_27/shapefiles/Obs_20260128_0059/Obs_area.shp")
specific_fire_shp = fire_shp_in[fire_shp_in['DSE_ID']==102636016]
#specific_fire_shp = fire_shp_in[fire_shp_in['CFA_ID']==260474]

specific_fire_shp = specific_fire_shp.to_crs(4326)
max_recalc_fbi.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
max_recalc_fbi.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
#Clip to the fire area:
clipped_recalc = max_recalc_fbi.rio.clip(specific_fire_shp.geometry.apply(mapping), specific_fire_shp.crs, drop=False)

clipped_recalc_ft = xr.merge([clipped_recalc, fuel_types])
fbi_table_ = clipped_recalc_ft.to_dataframe().dropna(subset='index_1')
fbi_table_ = fbi_table_.astype({'fuel_type': int})
"""
#load fuel lut for mapping on fuel categories:
path_to_fuel_lut_ = "C:/Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/fuel-type-model-authorised-vic-20250225011044.csv"
fuel_lut_ = pd.read_csv(path_to_fuel_lut_)
"""
fuel_FBM_dict = pd.Series(fuel_lut_.FBM.values,index=fuel_lut_.FTno_State).to_dict()
fbi_table_['FBM'] = fbi_table_['fuel_type'].map(fuel_FBM_dict)
fbi_table_ = fbi_table_.drop(columns=['band','spatial_ref', 'time']).reset_index()
#map lat and lon as shapely objects in geopandas:
geometry = [Point(xy) for xy in zip(fbi_table_['longitude'], fbi_table_['latitude'])]
fbi_gdf = geopandas.GeoDataFrame(fbi_table_, geometry=geometry)

plot_fbi_points_in_fire(fbi_gdf, specific_fire_shp)
"""
#%%
######################################
#Download observed weather.
#######################################
print('******************************')
print('Getting observations from weather station archive')
"""
#For now - choose AWS locations. Future - could auto select nearest obs?

start_time_obs = datetime(year=year_sel_,month=mon_sel,day=day_sel,hour=0,minute=0,second=0)
end_time_obs = datetime(year=year_sel_, month=mon_sel,day=day_sel,hour=23,minute=55,second=59)
"""

start_time_obs = datetime(year=year_sel_,month=mon_sel,day=day_sel,hour=8,minute=0,second=0)
end_time_obs = datetime(year=year_sel_, month=mon_sel,day=day_sel+1,hour=7,minute=55,second=59)

stations_to_pick = ["ALBURY AIRPORT AWS"]
#stations_to_pick = ["HUNTERS HILL"]

obs_table = make_observation_table_from_archive(start_time_obs, end_time_obs, stations_to_pick)
print(str(len(stations_to_pick))+' stations chosen in this calc')
print(str(len(obs_table['bom_id'].unique()))+' stations found.')
#Fix timestamps to end in every 10 minutes by subtracting whatever the last digit in the minute is
obs_table.time = obs_table.time - timedelta(minutes=int(str(obs_table.time.iloc[1].minute)[-1]))
obs_table.to_csv(outpath / 'obs_albury_20260108.csv')

#%%

#####################################
#Alternative: Load from alternative source, e.g. Tarnook download
#####################################
"""
#table_in = pd.read_csv("C:/Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data/PAWS_Delta-fts-data_2025-12-01T10_08_00.000Z_2026-02-16T23_08_00.000Z.csv", parse_dates=['Date'], date_format='mixed', dayfirst=True)
table_in = pd.read_csv("C:/Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data/obs_albury_20260105.csv", parse_dates=['time'], date_format='mixed', dayfirst=True)
#table_in2 = pd.read_csv("C:/Users/clark/analysis1/Case_studies/2026_01_09/compiled_weather_data/obs_albury_20260110.csv", parse_dates=['time'], date_format='mixed', dayfirst=True)


obs_table = table_in
"""
"""
obs_table = table_in.iloc[0::2]
obs_table['Date'] = obs_table['Date'].dt.tz_convert('Australia/Melbourne').dt.tz_localize(None)   #Time zone convert from UTC to local time, then remove TZ info
obs_table = pd.concat([table_in, table_in2])
obs_table = obs_table.drop(columns=['Unnamed: 0'])
obs_table = obs_table[(obs_table['Date'] >= start_time_obs) & (obs_table['Date'] <= end_time_obs)]
obs_table.reset_index(drop=True, inplace=True)

#wind conversion for the NSW PAWS
obs_table['wind speed kmh'] = obs_table['WSav600gnd'] * 1.825 * (np.log(10/0.2)/np.log(3/0.2))
obs_table['wind gust kmh'] = obs_table['WSmx600gnd'] * 1.825 * (np.log(10/0.2)/np.log(3/0.2))

#rename variables to be consistent with rest of code
obs_table = obs_table.rename(columns={'T': 'temperature',
                                      'H': 'RH',
                                      'WDav600gnd': 'wind dir deg',
                                      'Date': 'time'
                                })
#infill DF
obs_table['DF'] = np.full(len(obs_table), 9.1)  #use 9.1 same as Albury
"""

#%%
###################################
#Add column to convert wind dir to bearing
###################################

bearing_dict = {'N': 0.,
                'NNE': 22.5,
                'NE': 45.,
                'ENE': 67.5,
                'E': 90.,
                'ESE': 112.5,
                'SE': 135.,
                'SSE': 157.5,
                'S': 180.,
                'SSW': 202.5,
                'SW': 225.,
                'WSW': 247.5,
                'W': 270.,
                'WNW': 292.5,
                'NW': 315.,
                'NNW': 337.5}

obs_table['wind dir deg'] = obs_table['wind dir']
obs_table['wind dir deg'] = obs_table['wind dir deg'].map(bearing_dict)

#%%
#################################
#Calculate extra FBIs for local fuel types.
#################################

#ft_list = fbi_gdf['fuel_type'].unique()
#ft_list = [3046, 3049, 3050, 3048, 3028]
ft_list = [3020, 3007]
fbi_calcd_cols = calculate_fbi_for_fueltype(obs_table, fuel_lut_, ft_list)
gfdi_calcd = calculate_gfdi_for_awsdata(obs_table)
ffdi_calcd = calculate_ffdi_for_awsdata(obs_table)

obs_table = pd.concat([obs_table, fbi_calcd_cols], axis=1)
obs_table['FFDI'] = ffdi_calcd
obs_table['GFDI'] = gfdi_calcd

obs_table.to_csv(outpath / 'obs_fbicalc_eildonft_20260223.csv')
#%%
####################################
#Compare grids to observations.
#Do so for FBI, FFDI, GFDI, temp, wind speed and AFDRS ROS.
####################################

#Actually just grab the first pixel for each fuel type. For majority of fires this will be ok;
#but will need to adjust for highly complex terrain.

fig, axs = plt.subplots(figsize=(6,6))
fig_temp, axs_temp = plt.subplots(figsize=(6,6))
fig_rh, axs_rh = plt.subplots(figsize=(6,6))
fig_wind, axs_wind = plt.subplots(figsize=(6,6))
fig_ros, axs_ros = plt.subplots(figsize=(6,6))
fig_ffdi, axs_ffdi = plt.subplots(figsize=(6,6))
fig_gfdi, axs_gfdi = plt.subplots(figsize=(6,6))
#obs_table_trimmed = obs_table[obs_table['time'].dt.date==datetime_sel.date()]
obs_table_trimmed = obs_table

#Optional (comment in or out as needed): Set up calc for point forecast data.
#So within the loop we plot the FBI and ROS at the AWS location.
lat_sel = obs_table['latitude'].iloc[0]
lon_sel = obs_table['longitude'].iloc[0]
fc_table = recalc_file_in.sel(latitude=lat_sel, longitude=lon_sel, method='nearest').to_dataframe().iloc[start_ind:end_ind+1].reset_index()
fc_table['time'] = fc_table['time']+timedelta(hours=11)  #convert to AEDT
fc_table = fc_table.rename(columns={'T_SFC': 'temperature',
                            'RH_SFC': 'RH',
                            'Td_SFC': 'dew point',
                            'KBDI_SFC': 'KBDI',
                            'DF_SFC': 'DF',
                            'WindMagKmh_SFC': 'wind speed kmh',
                            'precipitation': 'accum precip',
                            'GrassFuelLoad_SFC': 'grass fuel load',
                            }
                           )
calcd_fc_cols = calculate_fbi_for_fueltype(fc_table, fuel_lut_, ft_list)
calcd_fc_cols.index = fc_table['time']-timedelta(hours=11)  #convert back to UTC to fit the loop. It's lazy, I know.

color_list = ('mediumblue', 'cornflowerblue', 'k', 'grey', 'darkred', 'darkorange', 'darkgreen', 'darkviolet')
i=0

for ft in ft_list:
#    if ft in [3050, 3046, 3051]:
#        continue
    """
    #Block for getting forecast at fire location:   
    lat_sel = fbi_gdf[fbi_gdf['fuel_type']==ft].iloc[0]['latitude']
    lon_sel = fbi_gdf[fbi_gdf['fuel_type']==ft].iloc[0]['longitude']
    
    fbi_at_fire = recalc_file_in['index_1'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()
    ros_at_fire = recalc_file_in['rate_of_spread'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()
    """
    #Block for getting forecast at AWS location:
    fbi_at_fire = calcd_fc_cols['FBI_'+str(ft)]
    ros_at_fire = calcd_fc_cols['ROS_'+str(ft)]
    
    im1 = axs.plot(fbi_at_fire.index+timedelta(hours=11), fbi_at_fire, color=color_list[i], linestyle='dashed')
    im2 = axs.plot(obs_table_trimmed['time'], obs_table_trimmed['FBI_'+str(ft)], color=color_list[i], label=str(ft))
    axs.hlines(y=12, xmin=obs_table_trimmed['time'].iloc[0], xmax=obs_table_trimmed['time'].iloc[-1], linewidth=2, color='green')
    axs.hlines(y=24, xmin=obs_table_trimmed['time'].iloc[0], xmax=obs_table_trimmed['time'].iloc[-1], linewidth=2, color='gold')
    axs.hlines(y=50, xmin=obs_table_trimmed['time'].iloc[0], xmax=obs_table_trimmed['time'].iloc[-1], linewidth=2, color='darkorange')
    im9 = axs_ros.plot(ros_at_fire.index+timedelta(hours=11), ros_at_fire, color=color_list[i], linestyle='dashed')
    im10 = axs_ros.plot(obs_table_trimmed['time'], obs_table_trimmed['ROS_'+str(ft)], color=color_list[i], label=str(ft))
    
    if i==0:
        fc_columns = pd.DataFrame({'FBI_'+str(ft)+'_FC': fbi_at_fire['index_1'].values, 'ROS_'+str(ft)+'_FC':ros_at_fire['rate_of_spread'].values})
        #fc_columns = pd.DataFrame({'FBI_'+str(ft)+'_FC': fbi_at_fire.values, 'ROS_'+str(ft)+'_FC':ros_at_fire.values})
    else:
        
        fc_columns['FBI_'+str(ft)+'_FC'] = fbi_at_fire['index_1'].values
        fc_columns['ROS_'+str(ft)+'_FC'] = ros_at_fire['rate_of_spread'].values
        """        
        fc_columns['FBI_'+str(ft)+'_FC'] = fbi_at_fire.values
        fc_columns['ROS_'+str(ft)+'_FC'] = ros_at_fire.values
        """
    i=i+1


#get variables at selected location (of fire or AWS location - depending on above chosen blocks)
temp_at_fire = recalc_file_in['T_SFC'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()
rh_at_fire = recalc_file_in['RH_SFC'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()
wind_at_fire = recalc_file_in['WindMagKmh_SFC'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()
ffdi_at_fire = recalc_file_in['FFDI_SFC'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()
gfdi_at_fire = recalc_file_in['GFDI_SFC'].sel(latitude=lat_sel, longitude=lon_sel, method='nearest')[start_ind:end_ind+1].to_dataframe()

#formatting parameters - set up windows for lower and upper y lims of plots
temp_upper = np.nanmax([30, max(obs_table_trimmed['temperature']+2), max(temp_at_fire['T_SFC']+2)])
temp_lower = np.nanmin([10, min(obs_table_trimmed['temperature']-2), min(temp_at_fire['T_SFC']-2)])
rh_lower = np.nanmin([20, min(obs_table_trimmed['RH']-5), min(rh_at_fire['RH_SFC']-5)])
rh_lower = np.nanmax([0, rh_lower])
rh_upper = np.nanmax([60, max(obs_table_trimmed['RH']+5), max(rh_at_fire['RH_SFC']+5)])
rh_upper = np.nanmin([100, rh_upper])
fbi_upper = np.nanmax([50, fc_columns[[col for col in fc_columns if col.startswith('FBI')]].max().max()+5,
                    obs_table_trimmed[[col for col in obs_table_trimmed if col.startswith('FBI')]].max().max()+5])  #a convoluted-ish way of saying "columns that have FBI in them, what's the max?"
fbi_lower = 0
wind_upper = np.nanmax([40, max(obs_table_trimmed['wind gust kmh']+10), max(wind_at_fire['WindMagKmh_SFC']+5)])
wind_lower = 0
winddir_upper = 360

#do the same for x lims
time_lower = obs_table_trimmed['time'].min()
time_upper = obs_table_trimmed['time'].max()

#plotting block - other plots
im3 = axs_temp.plot(temp_at_fire.index+timedelta(hours=11), temp_at_fire['T_SFC'], color='k', linestyle='dashed', label='forecast')
im4 = axs_temp.plot(obs_table_trimmed['time'], obs_table_trimmed['temperature'], color='k', label='obs')
im5 = axs_rh.plot(rh_at_fire.index+timedelta(hours=11), rh_at_fire['RH_SFC'], color='k', linestyle='dashed', label='forecast')
im6 = axs_rh.plot(obs_table_trimmed['time'], obs_table_trimmed['RH'], color='k', label='obs')
im7 = axs_wind.plot(wind_at_fire.index+timedelta(hours=11), wind_at_fire['WindMagKmh_SFC'], color='k', linestyle='dashed', label='forecast')
im8 = axs_wind.plot(obs_table_trimmed['time'], obs_table_trimmed['wind speed kmh'], color='k',  label='Wind speed')
im8a = axs_wind.plot(obs_table_trimmed['time'], obs_table_trimmed['wind gust kmh'], color='cornflowerblue', linestyle='dashed', label='Gust')

#axs_wind1 = axs_wind.twinx()
#im8b = axs_wind1.plot(obs_table_trimmed['time'], obs_table_trimmed['wind dir deg'], color='grey', linestyle='dotted', label='Wind dir')
im11 = axs_ffdi.plot(ffdi_at_fire.index+timedelta(hours=11), ffdi_at_fire['FFDI_SFC'], color='k', linestyle='dashed', label='forecast')
im12 = axs_ffdi.plot(obs_table_trimmed['time'], obs_table_trimmed['FFDI'], color='k', label='obs')
im13 = axs_gfdi.plot(gfdi_at_fire.index+timedelta(hours=11), gfdi_at_fire['GFDI_SFC'], color='k', linestyle='dashed', label='forecast')
im14 = axs_gfdi.plot(obs_table_trimmed['time'], obs_table_trimmed['GFDI'], color='k', label='obs')

axs.legend(fontsize=14)
axs.set_title('FBI', fontsize=18)
axs.set_xticklabels(axs.get_xticklabels(), rotation=90, fontsize=14)
axs.set_ylim(fbi_lower, fbi_upper)
axs.set_ylabel('FBI', fontsize=14)
axs.set_yticklabels(axs.get_yticklabels(), fontsize=14)
axs.set_xlim(time_lower, time_upper)
axs_temp.set_title('Temperature', fontsize=18)
axs_temp.set_xticklabels(axs_temp.get_xticklabels(),rotation=90, fontsize=14)
axs_temp.set_ylim(temp_lower, temp_upper)
axs_temp.set_ylabel('Temperature (deg C)', fontsize=14)
axs_temp.legend(fontsize=14)
axs_temp.set_yticklabels(axs_temp.get_yticklabels(), fontsize=14)
axs_temp.set_xlim(time_lower, time_upper)
axs_rh.set_title('Relative Humidity', fontsize=18)
axs_rh.set_xticklabels(axs_rh.get_xticklabels(), rotation=90, fontsize=14)
axs_rh.set_ylim(rh_lower, rh_upper)
axs_rh.set_ylabel('Rel. humidity (%)', fontsize=14)
axs_rh.set_yticklabels(axs_rh.get_yticklabels(), fontsize=14)
axs_rh.legend(fontsize=14)
axs_rh.set_xlim(time_lower, time_upper)
#axs_wind.set_title('Wind Speed', fontsize=18)
#axs_wind.set_title(stations_to_pick[0].title(), fontsize=18)
axs_wind.set_xticklabels(axs_wind.get_xticklabels(), rotation=90, fontsize=14)
axs_wind.set_ylim(wind_lower, wind_upper)
axs_wind.set_ylabel('Wind speed (km/h)', fontsize=14)
axs_wind.set_yticklabels(axs_wind.get_yticklabels(), fontsize=14)
axs_wind.set_xlim(time_lower, time_upper)
x0 = md.date2num(obs_table_trimmed['time'])  #convert to numbers which refers to days since... well, something...
x0 = x0[0::6]  #grab every 2nd value to avoid overcrowding
y0 = np.zeros(len(x0)) + 5
#y0 = obs_table_trimmed['wind gust kmh'].values -5
#y0 = y0[0::6]
dx = -np.sin(np.pi/180.*obs_table_trimmed['wind dir deg'].values) * 0.05
dx = dx[0::6]
dy = -np.cos(np.pi/180. * obs_table_trimmed['wind dir deg'].values) * 5
dy = dy[0::6]
y0 = y0 - 0.5* dy
for i in range(0, len(x0)):
    arr = plt.Arrow(x0[i], y0[i], dx[i], dy[i], width=0.03, edgecolor='black', color='black')
    axs_wind.add_patch(arr)
axs_wind.hlines(0, time_lower, time_upper, color='black')
"""
axs_wind1.set_ylabel('Wind dir', fontsize=14)
axs_wind1.set_ylim(0,360)
axs_wind1.set_yticks(np.arange(0,361,45))
axs_wind1.set_yticklabels(axs_wind1.get_yticklabels(), fontsize=14)
"""
ims = im7+im8+im8a
labels_wind = [l.get_label() for l in ims]
axs_wind.legend(ims, labels_wind, loc=0, fontsize=14)
axs_ros.legend(fontsize=14)
axs_ros.set_title('Rate of spread', fontsize=18)
axs_ros.set_xticklabels(axs_ros.get_xticklabels(), rotation=90, fontsize=14)
axs_ros.set_ylabel('Rate of spread (m/h)', fontsize=14)
axs_ros.set_yticklabels(axs_ros.get_yticklabels(), fontsize=14)
axs_ffdi.legend(fontsize=14)
axs_ffdi.set_title('FFDI', fontsize=18)
axs_ffdi.set_xticklabels(axs_ffdi.get_xticklabels(), rotation=90, fontsize=14)
axs_ffdi.set_ylabel('FFDI', fontsize=14)
axs_ffdi.set_yticklabels(axs_ffdi.get_yticklabels(), fontsize=14)
axs_gfdi.legend(fontsize=14)
axs_gfdi.set_title('GFDI', fontsize=18)
axs_gfdi.set_xticklabels(axs_gfdi.get_xticklabels(), rotation=90, fontsize=14)
axs_gfdi.set_ylabel('GFDI', fontsize=14)
axs_gfdi.set_yticklabels(axs_gfdi.get_yticklabels(), fontsize=14)


#fig.savefig(outpath / 'fbi_albury_walwa_20260109')
#fig_temp.savefig(outpath / 'temp_huntershill_walwa_20260105')
#fig_rh.savefig(outpath / 'rh_huntershill_walwa_20260105')
#fig_wind.savefig(outpath / 'wind_huntershill_walwa_20260105')
#fig_ros.savefig(outpath / 'ros_albury_walwa_20260109')
#fig_ffdi.savefig(outpath / 'ffdi_huntershill_walwa_20260105')
#fig_gfdi.savefig(outpath / 'gfdi_huntershill_walwa_20260105')
#%%
#Ruth Ryan wants it all in one panel stacked on one another. So I need an entirely new plotting block.
fig_p, axs_p = plt.subplots(4,1,figsize=(6,12))

axs_p[0].plot(temp_at_fire.index+timedelta(hours=11), temp_at_fire['T_SFC'], color='k', linestyle='dashed', label='forecast')
axs_p[0].plot(obs_table_trimmed['time'], obs_table_trimmed['temperature'], color='k', label='obs')
axs_p[1].plot(rh_at_fire.index+timedelta(hours=11), rh_at_fire['RH_SFC'], color='k', linestyle='dashed', label='forecast')
axs_p[1].plot(obs_table_trimmed['time'], obs_table_trimmed['RH'], color='k', label='obs')
axs_p[2].plot(wind_at_fire.index+timedelta(hours=11), wind_at_fire['WindMagKmh_SFC'], color='k', linestyle='dashed', label='forecast')
axs_p[2].plot(obs_table_trimmed['time'], obs_table_trimmed['wind speed kmh'], color='k',  label='Wind speed')
axs_p[2].plot(obs_table_trimmed['time'], obs_table_trimmed['wind gust kmh'], color='cornflowerblue', linestyle='dashed', label='Gust')
axs_p[3].plot(ffdi_at_fire.index+timedelta(hours=11), ffdi_at_fire['FFDI_SFC'], color='k', linestyle='dashed', label='forecast')
axs_p[3].plot(obs_table_trimmed['time'], obs_table_trimmed['FFDI'], color='k', label='obs')

axs_p[0].legend()
axs_p[0].set_xticklabels([])
axs_p[0].set_ylim(temp_lower, temp_upper)
axs_p[0].set_ylabel('Temperature (deg C)', fontsize=12)
axs_p[0].yaxis.set_tick_params(labelsize=12)
axs_p[0].set_xlim(time_lower, time_upper)
axs_p[1].legend()
axs_p[1].set_xticklabels([])
axs_p[1].set_ylim(rh_lower, rh_upper)
axs_p[1].set_ylabel('RH (%)', fontsize=12)
axs_p[1].yaxis.set_tick_params(labelsize=12)
axs_p[1].set_xlim(time_lower, time_upper)
axs_p[2].legend()
axs_p[2].set_xticklabels([])
axs_p[2].set_ylim(wind_lower, wind_upper)
axs_p[2].set_ylabel('Wind (km/h)', fontsize=12)
x0 = md.date2num(obs_table_trimmed['time'])  #convert to numbers which refers to days since... well, something...
x0 = x0[0::6]  #grab every 2nd value to avoid overcrowding
y0 = np.zeros(len(x0)) + 5
#y0 = obs_table_trimmed['wind gust kmh'].values -5
#y0 = y0[0::6]
dx = -np.sin(np.pi/180.*obs_table_trimmed['wind dir deg'].values) * 0.04
dx = dx[0::6]
dy = -np.cos(np.pi/180. * obs_table_trimmed['wind dir deg'].values) * 7
dy = dy[0::6]
y0 = y0 - 0.5* dy
for i in range(0, len(x0)):
    arr = plt.Arrow(x0[i], y0[i], dx[i], dy[i], width=0.02, edgecolor='black')
    axs_p[2].add_patch(arr)
axs_p[2].hlines(0, obs_table_trimmed['time'].min(), obs_table_trimmed['time'].max(), color='k')
axs_p[2].yaxis.set_tick_params(labelsize=12)
axs_p[2].set_xlim(time_lower, time_upper)
axs_p[3].legend()
axs_p[3].xaxis.set_tick_params(rotation=90, labelsize=12)
axs_p[3].set_ylabel('FFDI', fontsize=12)
axs_p[3].yaxis.set_tick_params(labelsize=12)
axs_p[3].set_xlim(time_lower, time_upper)

fig_p.suptitle('Albury Airport AWS', fontsize=16)

fig_p.tight_layout()
#fig_p.savefig( outpath / 'obspanel_albury_walwa_20260105')

#%%
#############################
#Save the hourly ROS - actual vs forecast.
#############################

#We actualy want the hour to be the midpoint of the averaging.
#So - average the 10 minutely obs by hour, using the hour timestep as the midpoint.
#With the pandas resample-mean method - it averages everything with the same "hour" value.
#So do a time shift of 30 mins - this means that the average is calculated on shifted hour value,
#which *was* timesteps that have the hour as the midpoint!

#SC edit 26/2/26: Add in wind direction: Closest to the hour.

obs_table_avg_for_saving = obs_table_trimmed.drop(columns=['bom_id', 'station_full','station_desc',
                                                          'latitude','longitude', 'wind gust kmh',
                                                          'upper_soil_fullness',
                                                          'accum precip', 'primary FBM', 'primary FBI',
                                                          'secondary FBM', 'secondary FBI'])
obs_table_avg_for_saving.index = obs_table_avg_for_saving['time']
obs_table_avg_for_saving.drop(columns='time', inplace=True)
#The way to do this is to pull out the direction as a separate variable,
#delete it from the final table so that the other variables can be sampled, 
#average the main variables (T, RH, FBI etc) on the hour using timeshift and resample,
#resample the direction (on that separate variable) on the hour,
#then add back in to the final table.
dir_series = obs_table_avg_for_saving['wind dir']
obs_table_avg_for_saving.drop(columns=['wind dir'], inplace=True)
obs_table_avg_for_saving = obs_table_avg_for_saving.shift(freq='30Min').resample('H').mean()
dir_series = dir_series.resample('H', origin='start').first()

#Add in the forecast data for comparison.
obs_table_avg_for_saving = obs_table_avg_for_saving.iloc[0:24]
obs_table_avg_for_saving['wind dir'] = dir_series.values
obs_table_avg_for_saving['Temperature_FC'] = temp_at_fire['T_SFC'].values
obs_table_avg_for_saving['RH_FC'] = rh_at_fire['RH_SFC'].values
obs_table_avg_for_saving['Wind_FC'] = wind_at_fire['WindMagKmh_SFC'].values
obs_table_avg_for_saving['FFDI_FC'] = ffdi_at_fire['FFDI_SFC'].values
obs_table_avg_for_saving['GFDI_FC'] = gfdi_at_fire['GFDI_SFC'].values

obs_table_output = obs_table_avg_for_saving[['temperature', 'Temperature_FC',
                                             'RH','RH_FC',
                                             'wind speed kmh', 'Wind_FC', 'wind dir', 'wind dir deg', 
                                             'KBDI','DF','curing','grass fuel load',
                                             'FFDI','FFDI_FC',
                                             'GFDI','GFDI_FC']]

obs_table_output = obs_table_output.rename({'temperature': 'Temperature_obs',
                                            'RH': 'RH_obs',
                                            'wind speed kmh': 'Wind_obs_kmh',
                                            'wind dir': 'Wind_dir_obs',
                                            'wind dir deg': 'Wind_deg_obs',
                                            'FFDI': 'FFDI_obs',
                                            'GFDI': 'GFDI_obs'})

for ft in ft_list:
    obs_table_output['FBI_'+str(ft)+'_obs'] = obs_table_avg_for_saving['FBI_'+str(ft)]
    obs_table_output['ROS_'+str(ft)+'_obs'] = obs_table_avg_for_saving['ROS_'+str(ft)]
    obs_table_output['FBI_'+str(ft)+'_FC'] = fc_columns['FBI_'+str(ft)+'_FC'].values
    obs_table_output['ROS_'+str(ft)+'_FC'] = fc_columns['ROS_'+str(ft)+'_FC'].values

#obs_table_output.to_csv(outpath / 'hourly_obs_comparison_mtbuller_20260111.csv')
#%%
#########################
#Tidy up, close:
#########################
recalc_file_in.close()