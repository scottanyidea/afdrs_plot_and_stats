#Extension of calc_kbdi_footprint_avg.py
#This is to calculate the entirety of the landscape risk matrix.

#WHAT WE NEED TO SET:
#Set date of forecast to use for KBDI and curing.
#Ensure the curing and KBDI are up to date.
#Check clim_data on BOM FTP and update to latest fortnightly file.
#Choose the fortnight we want.
#Check for updates to the bare soil fraction (ie. MODIS data)
#Update the link to the rainfall anomaly files.


import numpy as np
import xarray as xr
from pathlib import Path
import geopandas
from datetime import datetime, timedelta
from fbi_vic_plot_functions import plot_kbdi, plot_curing
import rioxarray
import pandas as pd
from shapely.geometry import mapping
import urllib.request
import shutil
from contextlib import closing

#Thresholds and scores for each of the input variables.
BARE_EARTH_ADJ = [0, -1, -2, -3]
CURING_THRESHOLDS = [65, 75, 85, 95]
RAINFALL_THRESHOLDS = [0, -30, -70, -110]
BARE_EARTH_THRESHOLDS = [20, 30, 40]
KBDI_THRESHOLDS = [75, 100, 125, 150]
AFDRS_ODDS_THRESHOLDS = [40, 60, 70, 80]




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

def obs_products_from_bom_ftp_urllib(product, downloaded_data_path):
    """
    ftp://ftp.bom.gov.au/
    IDZ20081_current_obs.json.gz which is on the BOM FTP site bom773 which TSU 
    user = "bomw0755", passwd = "pRee66mP"
    """
    url = "ftp://ftp.bom.gov.au/register/bom023/rKCm6dbQ/clim_data/" + product
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        with closing(resp) as r:
            with open(downloaded_data_path / product, 'wb') as f:
                shutil.copyfileobj(r, f)

if __name__=="__main__":
    #Set dates:
    year_sel_ = 2025
    mon_sel = 2
    day_sel = 24

    datetime_sel = datetime(year_sel_, mon_sel, day_sel)
    forecast_day_kbdi = 6
    
    #Set fortnight we want to look at for AFDRS median.
    #0 is the current fortnight. 1 starts a week later, and so forth - up to 3.
    afdrs_fn = 0

    #set strings here - bit of a mess but helps later!
    mon_sel_str = datetime.strftime(datetime_sel, "%m")
    day_sel_str = datetime.strftime(datetime_sel, "%d")
    
    ######################################################################
    #Load the datasets for the matrix:

    #KBDI and curing are currently produced from the latest forecasts, archived on the towers.
    
    #I use the archive version because it is at the AFDRS forecast resolution. It is possible to use the
    #current forecast from the BoM FTP for KBDI.
    #Using the current curing map from Tarnook or similar means a different resolution will be used, and regridding
    #will be necessary.
    if mon_sel <=6:
        fy_string = str(year_sel_-1)+"-"+str(year_sel_)
    else:
        fy_string = str(year_sel_)+"-"+str(year_sel_+1)
    print('Loading curing and KBDI')
#    kbdi_path = "M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/"+fy_string+"/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str
    kbdi_path = "C://Users/clark/analysis1/forecast_grids_alternative/"+fy_string+"/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str
    kbdi_in_file = xr.open_dataset(kbdi_path+"/IDV71147_VIC_KBDI_SFC.nc")
    curing_in_file = xr.open_dataset(kbdi_path+"/IDZ10148_AUS_FSE_curing_SFC.nc.gz")

    #Soil cover is calculated from a TIF layer produced from MODIS satellite data. 
    #This needs to be manually updated.
    #This also needs to be converted from tif to xarray for processing.
    print("loading bare soil cover TIF and converting")
    soil_cover_in = loadGeoTiff("C://Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/MODIS_BareSoil/MODIS_BareSoilCover_202412.tif")
    soil_cover_in = regrid_xr(kbdi_in_file, soil_cover_in)    

    
    #AFDRS chance of exceeding median:
    #This comes from the BoM 023 ftp server. I've chosen to use the fortnightly values.
    print("Downloading BoM AFDRS seasonal forecast data and opening")
    fc_product_name = 'IDCK000103.index_1.forecast.raw.median.aus.fortnightly.20250221.nc'
    DOWNLOADED_DATA_PATH = Path("C://Users/clark/analysis1/Bom_clim_fc_data")
    if (DOWNLOADED_DATA_PATH / fc_product_name).exists():
        print('Seasonal forecast data already exists locally. No need to download again.')
    else:
        obs_products_from_bom_ftp_urllib(fc_product_name, DOWNLOADED_DATA_PATH)
    chance_afdrs_median_file = xr.open_dataset(DOWNLOADED_DATA_PATH / fc_product_name)
    
    #This file contains 4 dimensions. The first co-ordinate just reverses the variables - we set to 1
    #for the chance of being ABOVE median AFDRS. 0 gives the chance of being below.
    #Second co-ordinate is the time window. 0 gives the current fortnight, 1 starts 1 week later,
    #2 starts 2 weeks later (so we have rolling fortnights).
    chance_afdrs_median = chance_afdrs_median_file['percentage_of_ensembles'][1,afdrs_fn,:,:]

    #################################################
    #Load the necessary shape files and masks:
    print('Loading necessary shapefiles')
    #Current ICC boundaries shapefile. I got sent this but this can be pulled from wherever.
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/ICC/ICC.shp")
    
    #Grass forest mask, as used to calculate FFDI/GFDI.
    #This needs to be converted from tif to xarray for processing.
    grass_forest_mask = loadGeoTiff("C://Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/v4_forest0grass1malleeheath1heathland0.tif")
    grass_forest_mask = regrid_xr(kbdi_in_file, grass_forest_mask)    

    
    ###########################
    #Load and process data.
    kbdi_in = kbdi_in_file['KBDI_SFC'][forecast_day_kbdi,:,:]
    curing_in = curing_in_file['grass_curing'][0,:,:]
    
    #Apply grass forest mask to make sure we get curing and bare soil cover only for grassy areas.
    print('Applying grass forest mask to curing and bare earth')
    curing_in = xr.where(grass_forest_mask==1, curing_in, np.nan)   
    soil_cover_in = xr.where(grass_forest_mask==1, soil_cover_in, np.nan)
    soil_cover_in = xr.where(soil_cover_in<=100, soil_cover_in, np.nan)


    #Set spatial dims and CRS for clipping to each ICC footprint:
    print('Setting spatial dims and CRS for data in order to clip to regions')
    soil_cover_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    soil_cover_in.rio.write_crs("EPSG:4326",inplace=True)
    curing_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    curing_in.rio.write_crs("EPSG:4326", inplace=True)
    kbdi_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    kbdi_in.rio.write_crs("EPSG:4326", inplace=True)
    chance_afdrs_median.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True)
    chance_afdrs_median.rio.write_crs("EPSG:4326", inplace=True)
    
    
    
    #Set up variables for storing:
    region_avg_cur = []
    region_avg_kbdi = []
    bare_earth_area = []
    afdrs_median_list = []
    rainfall_anom_list = []
    
    
    #Calculate spatial averages for each region:
    region_list = shp_in['ICC_NAME']
    #While inside the loop we want to grab the rainfall deficit. These are separate files, so I want to set the path here.
    rain_anomaly_path = Path("C:/Users/clark/OneDrive - Country Fire Authority/Documents - Fire Risk, Research & Community Preparedness - RD private/DATA/tmp/")
    
    #Loop through each region: Clip each variable, then calculate the mean. Then assign a score to the values.
    print('Looping through each region, clipping and calcing variables')
    for region in region_list:
        print('Starting '+region)
        area_polygon = shp_in[shp_in['ICC_NAME']==region]
        clipped_soil = soil_cover_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        clipped_kbdi = kbdi_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        clipped_cur = curing_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        clipped_afdrs = chance_afdrs_median.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
        region_avg_cur.append(np.nanmean(clipped_cur))
        region_avg_kbdi.append(np.nanmean(clipped_kbdi))
        bare_earth_area.append(np.nanmean(clipped_soil))
        afdrs_median_list.append(np.nanmean(clipped_afdrs))
                
        #While inside the loop: Calculate rainfall deficit.
        rainfile_name = "[ICC]["+region.upper()+"][RAINFALL][FIRE_SEASON][CUMULATIVE_DAILY_ANOMALY_CUMUL][07-01_TO_06-30].csv"
        rain_anom = pd.read_csv(rain_anomaly_path / "Rainfall_anomaly_20250220" / "ICC" / rainfile_name)
        rainfall_anom_list.append(rain_anom.iloc[rain_anom[fy_string].last_valid_index()][fy_string])

    #Produce rating for each variable in the list:
    print("Calculating risk scores")
    curing_rating = calc_thresholds(np.array(region_avg_cur), CURING_THRESHOLDS, dir='increasing')
    bareearth_rating = calc_bare_earth_adj(np.array(bare_earth_area), BARE_EARTH_THRESHOLDS)
    overall_grass_rating = curing_rating+bareearth_rating
    overall_grass_rating = np.where(overall_grass_rating<0, 0, overall_grass_rating)
    kbdi_rating = calc_thresholds(np.array(region_avg_kbdi), KBDI_THRESHOLDS, dir='increasing')
    afdrs_median_rating = calc_thresholds(np.array(afdrs_median_list), AFDRS_ODDS_THRESHOLDS, dir='increasing')
    rainfall_anom_rating = calc_thresholds(np.array(rainfall_anom_list), RAINFALL_THRESHOLDS, dir='decreasing')
    total_rating = overall_grass_rating + kbdi_rating + afdrs_median_rating + rainfall_anom_rating
    
    print("Building matrix and saving")
    region_matrix = pd.DataFrame(data={'ICC_Footprint':region_list, 
                                       'Curing':region_avg_cur, 'Curing_score':curing_rating,
                                       'Bare_earth_frac':bare_earth_area, 'Bare_earth_adjustment':bareearth_rating, 'Overall_grass': overall_grass_rating,
                                       'KBDI': region_avg_kbdi, 'KBDI_score': kbdi_rating,
                                       'Rainfall_anom':rainfall_anom_list, 'Rainfall_anom_score':rainfall_anom_rating,
                                       'AFDRS_chance_above_median': afdrs_median_list, 'AFDRS_score': afdrs_median_rating,
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
    
    region_matrix.to_csv('ICCFootprintMatrix_20250224.csv', index=False)
    chance_afdrs_median.close()
    kbdi_in_file.close()
    curing_in_file.close()
    