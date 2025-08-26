
#This is to calculate the entirety of the ICC footprint risk matrix for medium term forecasts.

#Update 18/8/25: This is v2, v1 is called "calc_landscape_risk_matrix.py"
#V2 uses a new set of inputs, and the eventual intention is for this to be completely
#automated.

#Update 26/8/25: 
#We've now decided to use 3 monthly rainfall deficits instead of soil moisture. This is now v2.1.

#INPUTS:
#Grassland curing - daily in ADFD

#AussieGrass (Long Paddock) pasture growth - downloaded from Long Paddock

#12 month rainfall percentiles - from AWAP data. Currently on Sharepoint, aim is to point to tower data
#Using available rainfall spatial averages by ICC footprint - calculate percentiles in this script,
#then rank the recent 12 months against the percentiles. This future proofs for updated data.

#Soil moisture (root zone) - from AWO, downloaded to tower, currently on Sharepoint

#Fortnightly seasonal forecasts from BoM - chance of unusually warm, and chance of above median rainfall
#Both of those directly downloaded from BoM FTP

import numpy as np
import xarray as xr
from pathlib import Path
import geopandas
from datetime import datetime, timedelta
import rioxarray
import pandas as pd
from shapely.geometry import mapping
import requests
import shutil
from contextlib import closing
from scipy import stats
from ftplib import FTP
import zipfile
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib.cm as cm

#Thresholds and scores for each of the input variables.
#These thresholds should be ordered in terms of score - so if 
#lower values = higher scores, order in decreasing values.
CURING_THRESHOLDS = [45, 60, 75, 85]
RAINFALL_THRESHOLDS = [70, 40, 20, 10]
SOIL_MOISTURE_THRESHOLDS = [0.7, 0.4, 0.1, 0.01]
TEMP_ODDS_THRESHOLDS = [10, 30, 60, 80]
RAINFALL_FC_THRESHOLDS = [60, 40, 30, 20]
PASTURE_GROWTH_THRESHOLDS = [30,70,80,90]



def calc_thresholds(values, thresholds, dir='increasing'):
    #setup output array
    rating = np.full(len(values), 0)
    
    #assign ratings based on thresholds
    if dir=='increasing':
        rating[values >= thresholds[3]] = 4
        rating[values < thresholds[3]] = 3
        rating[values < thresholds[2]] = 2
        rating[values < thresholds[1]] = 1
        rating[values < thresholds[0]] = 0
    elif dir=='decreasing':
        rating[values <= thresholds[3]] = 4
        rating[values > thresholds[3]] = 3
        rating[values > thresholds[2]] = 2
        rating[values > thresholds[1]] = 1
        rating[values > thresholds[0]] = 0
    else:
        raise TypeError("Dir must be 'increasing' or 'decreasing'. ") 
    return rating

def calc_bare_earth_adj(values, thresholds):
    adjustment = np.full(len(values), 0) 
    
    adjustment[values >= thresholds[2]] = -3
    adjustment[values < thresholds[2]] = -2
    adjustment[values < thresholds[1]] = -1
    adjustment[values < thresholds[0]] = 0
    
    return adjustment

def loadGeoTiff(tif_dir, da_var_name = None, as_Dataset = False):
    """
    Load a GeoTiff file into the xarray Dataset or DataArray type
    """
    _d = rioxarray.open_rasterio(tif_dir, parse_coordinates=True)

    if (as_Dataset):
        _d = _d.to_dataset('band')
        _d = _d.rename({1: da_var_name})
    else:
        _d = _d.rename(da_var_name)
        if len(_d.band > 1):    #From July 2023 (?), the fuel type GeoTIFF has 2 layers, we want only the first
            _d = _d[0,:,:]
        else:
            _d = _d.drop("band").squeeze('band')
    
    _d = _d.rename({'x': 'longitude', 'y': 'latitude'})   # Rename dimensions
    _d = _d.rio.set_spatial_dims(x_dim = 'longitude', y_dim = 'latitude')
    _d = _d.rio.write_crs("epsg:4326", inplace=True).rio.write_coordinate_system(inplace=True)

    return _d

def regrid_xr(xr_temp, xr_new, method = "nearest"):
    """
    Match up the dim/coord of two Dataset or DataArray objects using the interp function.
    """
    if ("latitude" in xr_temp.coords) and ("latitude" in xr_new.coords):
        return xr_new.interp(latitude = xr_temp.latitude, longitude = xr_temp.longitude, method=method)
    elif ("lat" in xr_temp.coords) and ("lat" in xr_new.coords):     
        return xr_new.interp(lat = xr_temp.lat, lon = xr_temp.lon, method=method)


if __name__=="__main__":
    #Set dates:
    year_sel_ = 2025
    mon_sel = 1
    day_sel = 12

    datetime_sel = datetime(year_sel_, mon_sel, day_sel)
    

    #set strings here - bit of a mess but helps later!
    mon_sel_str = datetime.strftime(datetime_sel, "%m")
    day_sel_str = datetime.strftime(datetime_sel, "%d")
    
    ######################################################################
    #Load necessary shapefiles:
    print('Loading necessary shapefiles')
    #Current ICC boundaries shapefile. I got sent this but this can be pulled from wherever.
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/ICC/ICC.shp")
    region_list = shp_in['ICC_NAME']
    
    #Grass forest mask, as used to calculate FFDI/GFDI.
    #This needs to be converted from tif to xarray for processing.
    grass_forest_mask = loadGeoTiff("C://Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/v4_forest0grass1malleeheath1heathland0.tif")
    
    ######################################################################
    #Load the datasets for the matrix:
    
    ###CURRENT STATE###    
    
    #Curing - we obtain from ADFD grids - point to archive on network drive for now
    if mon_sel <=6:
        fy_string = str(year_sel_-1)+"-"+str(year_sel_)
    else:
        fy_string = str(year_sel_)+"-"+str(year_sel_+1)
    print('Loading curing')
    adfd_path = "M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/"+fy_string+"/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str
#    kbdi_path = "C://Users/clark/analysis1/forecast_grids_alternative/"+fy_string+"/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str
    curing_in_file = xr.open_dataset(adfd_path+"/IDZ10148_AUS_FSE_curing_SFC.nc.gz")

    #PLaceholder for grass fuel load product - current state.
    #As interim we use AussieGrass pasture growth product - most recent month.
    pg_path = "C://Users/clark/analysis1/afdrs_fbi_recalc/data/"
    #pg_pctile = loadGeoTiff(pg_path+'/202507.01months.growth.pcnt.aus.tiff')
    pg_url = 'https://data.longpaddock.qld.gov.au/AussieGRASS2/AUS/202507/202507.01months.growth.pcnt.aus.tiff'
    resp = requests.get(pg_url)
    with open(pg_path+'pg_data_current.tiff','wb') as f:
        f.write(resp.content)
    pg_pctile = loadGeoTiff(pg_path+'pg_data_current.tiff')
    pg_pctile = xr.where(pg_pctile>100, np.nan, pg_pctile)
    
    
    #Past 12 months rainfall decile:
    #This comes from tower 2
    #But interim measure is to use the temporary rainfall archive on sharepoint
    print('Loading rainfall and calculating deciles')
    region_rain_pctiles_12m = []
    region_rain_pctiles_3m = []
    ICC_RAINFALL_PATH = Path("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/tmp/BPS/Output/Rainfall/ICC/IMG/")
    for region in region_list:
        fc_product_name = '[ICC]['+region+'][RAINFALL][ANNUAL_YEAR][SPATIALLY_AVERAGED].csv'
        icc_rainfall_avg = pd.read_csv(ICC_RAINFALL_PATH / fc_product_name, parse_dates=['time'])
    
        #Calculate monthly deciles:
        #First need to sum to monthly totals
        rain_grouped = icc_rainfall_avg.groupby([icc_rainfall_avg['time'].dt.year, icc_rainfall_avg['time'].dt.month]).sum('RAIN')
        rain_grouped.index.rename(('year','month'), inplace=True)
        rain_grouped = rain_grouped['RAIN']
    
        #Calculate the yearly totals for every 12 month block (start/ending month prior to current month),
        #from 1910 to current - this produces the annual rainfall record to calculate percentiles
        totals_12m = []
        totals_3m = []
        for yr in range(icc_rainfall_avg.Year.min(), icc_rainfall_avg.Year.max()):
            totals_12m.append(
                rain_grouped[((rain_grouped.index.get_level_values('year') == yr+1) & (rain_grouped.index.get_level_values('month') < mon_sel)) 
                         | ((rain_grouped.index.get_level_values('year') == yr) & (rain_grouped.index.get_level_values('month') >=mon_sel)) ].sum()
                )
            if mon_sel >=4:
                totals_3m.append(
                    rain_grouped[((rain_grouped.index.get_level_values('year') == yr+1) & (rain_grouped.index.get_level_values('month') < mon_sel)) 
                                 | ((rain_grouped.index.get_level_values('year') == yr+1) & (rain_grouped.index.get_level_values('month') >=mon_sel-3)) ].sum()
                    )
            else:
                #case to capture area around start of year
                totals_3m.append(
                    rain_grouped[((rain_grouped.index.get_level_values('year') == yr+1) & (rain_grouped.index.get_level_values('month') < mon_sel)) 
                                 | ((rain_grouped.index.get_level_values('year') == yr) & (rain_grouped.index.get_level_values('month') >=9+mon_sel)) ].sum()
                    )
        #The desired percentile score is just the last value. Calculate the percentile score.
        region_rain_pctiles_12m.append(stats.percentileofscore(totals_12m, totals_12m[-1]))
        region_rain_pctiles_3m.append(stats.percentileofscore(totals_3m, totals_3m[-1]))
    
    
    ##########################
    #FUTURE STATE:
    print('Downloading climate forecast data from BoM')
    #Set destination:
    bom_fc_dest_path = "C:/Users/clark/analysis1/Bom_clim_fc_data/"
    #Product name - this one includes rainfall and temp extremes forecasts
    product_name = 'IDCK000098.zip'
    
    
    #log in to FTP and download products:
    ftp = FTP("ftp.bom.gov.au")
    ftp.login(user='bom604', passwd='tiventer')
    ftp.cwd('clim_data/')
    with open(bom_fc_dest_path+"/"+product_name, 'wb') as fobj:
        ftp.retrbinary('RETR %s' % product_name, fobj.write)

    #Unzip and load data:
    with zipfile.ZipFile(bom_fc_dest_path+'/'+product_name) as z:
        #Get the list of files in the zip archive - then just get the fortnightly chance of top 20% temperatures.
        #This is to allow date to be arbitrary (there's normally only one timestamp, which updates daily)
        file_list = z.namelist()
        fn_file = fnmatch.filter(file_list, 'tmax.forecast.calib.quintile_top.aus.fortnightly.*.nc')
        with z.open(fn_file[0]) as zf, open(bom_fc_dest_path+'/'+fn_file[0], 'wb') as f:
            shutil.copyfileobj(zf, f)
    tmax_fc_in = xr.open_dataset(bom_fc_dest_path+'/'+fn_file[0])
        
    #Download other product - chance of above median rainfall:
    product_name = 'IDCKO6MCFM.zip'
    with open(bom_fc_dest_path+"/"+product_name, 'wb') as fobj:
        ftp.retrbinary('RETR %s' % product_name, fobj.write)
    ftp.quit()
    
    #Unzip and load data:
    with zipfile.ZipFile(bom_fc_dest_path+'/'+product_name) as z:
        file_list = z.namelist()
        fn_file = fnmatch.filter(file_list, 'rain.forecast.raw.median.aus.fortnightly.*.nc')
        with z.open(fn_file[0]) as zf, open(bom_fc_dest_path+'/'+fn_file[0], 'wb') as f:
            shutil.copyfileobj(zf, f)
    rain_forecast_in = xr.open_dataset(bom_fc_dest_path+'/'+fn_file[0])
    

    
    ###########################
    #Load and process data.
    curing_in = curing_in_file['grass_curing'][0,:,:]
    tmax_fc = tmax_fc_in['probquintile_five'][0,:,:]
    rain_fc = rain_forecast_in['probF'][0,:,:]
    
    #Apply grass forest mask to make sure we get curing and bare soil cover only for grassy areas.
    print('Applying grass forest mask to curing and bare earth')
    grass_forest_mask = regrid_xr(curing_in, grass_forest_mask)
    curing_in = xr.where(grass_forest_mask==1, curing_in, np.nan)   
    

    #Set spatial dims and CRS for clipping to each ICC footprint:
    print('Setting spatial dims and CRS for data in order to clip to regions')
    curing_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    curing_in.rio.write_crs("EPSG:4326", inplace=True)
    tmax_fc.rio.set_spatial_dims(x_dim='lon',y_dim='lat',inplace=True)
    tmax_fc.rio.write_crs("EPSG:4326", inplace=True)
    rain_fc.rio.set_spatial_dims(x_dim='lon',y_dim='lat',inplace=True)
    rain_fc.rio.write_crs("EPSG:4326", inplace=True)
    pg_pctile.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude', inplace=True)
    pg_pctile.rio.write_crs("EPSG:4326", inplace=True)

    #Set up variables for storing:
    region_avg_cur = []
    region_avg_tmax_fc = []
    region_avg_rain_fc = []
    region_avg_pg = []
    
    #Calculate spatial averages for each region:

    #While inside the loop we want to grab the rainfall deficit. These are separate files, so I want to set the path here.
    #Loop through each region: Clip each variable, then calculate the mean. Then assign a score to the values.
    print('Looping through each region, clipping and calcing variables')
    for region in region_list:
        print('Starting '+region)
        area_polygon = shp_in[shp_in['ICC_NAME']==region]
        clipped_cur = curing_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        clipped_pg = pg_pctile.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        region_avg_cur.append(np.nanmean(clipped_cur))
        region_avg_pg.append(np.nanmean(clipped_pg))
        clipped_tmaxfc = tmax_fc.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        clipped_rainfc = rain_fc.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        region_avg_tmax_fc.append(np.nanmean(clipped_tmaxfc))
        region_avg_rain_fc.append(np.nanmean(clipped_rainfc))

    #Produce rating for each variable in the list:
    print("Calculating risk scores")
    curing_rating = calc_thresholds(np.array(region_avg_cur), CURING_THRESHOLDS, dir='increasing')
    grass_fl_rating = calc_thresholds(np.array(region_avg_pg), PASTURE_GROWTH_THRESHOLDS, dir='increasing')
    rainfall_pctile_rating_3m = calc_thresholds(np.array(region_rain_pctiles_3m), RAINFALL_THRESHOLDS, dir='decreasing')
    rainfall_pctile_rating_12m = calc_thresholds(np.array(region_rain_pctiles_12m), RAINFALL_THRESHOLDS, dir='decreasing')
    rainfall_fc_rating = calc_thresholds(np.array(region_avg_rain_fc), RAINFALL_FC_THRESHOLDS, dir='decreasing')
    temp_fc_rating = calc_thresholds(np.array(region_avg_tmax_fc), TEMP_ODDS_THRESHOLDS, dir='increasing')
    total_rating = curing_rating + grass_fl_rating + rainfall_pctile_rating_12m + rainfall_pctile_rating_12m + temp_fc_rating + rainfall_fc_rating
    
    print("Building matrix and saving")
    region_matrix = pd.DataFrame(data={'ICC_Footprint':region_list, 
                                       'Curing':region_avg_cur, 'Curing_score':curing_rating,
                                       'Grass FL score': grass_fl_rating,
                                       '3 month rain pctile': region_rain_pctiles_3m, '3 month rain score': rainfall_pctile_rating_3m,
                                       '12 month rain pctile': region_rain_pctiles_12m, '12 month rain score': rainfall_pctile_rating_12m,
                                       'Chance temp extremes':region_avg_tmax_fc, 'Temp extreme score':temp_fc_rating,
                                       'Chance rain above median': region_avg_rain_fc, 'Rain fc score': rainfall_fc_rating,
                                       'Overall_score': total_rating}
                                 )

    region_matrix['Region'] = region_matrix['ICC_Footprint'].map({'Colac':'Barwon South West',
                                                                  'Geelong':'Barwon South West',
                                                                  'Heywood':'Barwon South West',
                                                                  'Warrnambool':'Barwon South West',
                                                                  'Ferntree Gully':'East Metro',
                                                                  'Woori Yallock':'East Metro',
                                                                  'Bairnsdale':'Gippsland',
                                                                  'Bendoc':'Gippsland',
                                                                  'Erica':'Gippsland',
                                                                  'Heyfield':'Gippsland',
                                                                  'Noojee':'Gippsland',
                                                                  'Swifts Creek':'Gippsland',
                                                                  'Orbost': 'Gippsland',
                                                                  'Warragul':'Gippsland',
                                                                  'Ararat':'Grampians',
                                                                  'Ballarat':'Grampians',
                                                                  'Horsham':'Grampians',
                                                                  'Alexandra':'Hume',
                                                                  'Corryong':'Hume',
                                                                  'Mansfield':'Hume',
                                                                  'Ovens':'Hume',
                                                                  'Seymour':'Hume',
                                                                  'Shepparton': 'Hume',
                                                                  'Tallangatta': 'Hume',
                                                                  'Wangaratta':'Hume',
                                                                  'Wodonga':'Hume',
                                                                  'Bendigo':'Loddon Mallee',
                                                                  'Gisborne':'Loddon Mallee',
                                                                  'Mildura':'Loddon Mallee',
                                                                  'Sunshine':'North West Metro',
                                                                  'Dandenong':'South West Metro',})
    region_list2 = region_matrix['Region']
    region_matrix.drop(labels=['Region'], axis=1, inplace=True)
    region_matrix.insert(0, 'Region', region_list2)
    region_matrix = region_matrix.sort_values(by=['Region','ICC_Footprint'], ascending=[True,True])
    
    region_matrix.to_csv('ICCFootprintMatrix_test.csv', index=False)
    tmax_fc_in.close()
    rain_forecast_in.close()
    curing_in_file.close()
    
    shp_in['Score'] = total_rating
    
    #%%
    fig, axs = plt.subplots(1)
    cmap_rating = pltcolors.ListedColormap(['lightskyblue','antiquewhite','gold','darkorange','indianred'])
    norm = pltcolors.BoundaryNorm([0,4,8,12,16,20], cmap_rating.N)
    shp_in.plot(ax=axs, column='Score', edgecolor='black', linewidth=0.4, cmap=cmap_rating, norm=norm)
    fig.colorbar(cm.ScalarMappable(cmap=cmap_rating, norm=norm), ax=axs, orientation='vertical', fraction=0.058)
    axs.set_title('ICC outlook risk score')
