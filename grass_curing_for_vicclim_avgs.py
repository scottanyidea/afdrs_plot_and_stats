#Calculate spatially averaged grass curing from the inputs for VicClim that
#Bart created.

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import geopandas
import rioxarray
from shapely.geometry import mapping

#Below just copy paste from helper functions in afdrs recalc, i couldn't load it directly 
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
        
        _d = _d.rename({'x': 'lon', 'y': 'lat'})   # Rename dimensions
        _d = _d.rio.set_spatial_dims(x_dim = 'lon', y_dim = 'lat')
        _d = _d.rio.write_crs("epsg:4326", inplace=True).rio.write_coordinate_system(inplace=True)

        return _d

def preprocess_forest_grass_mask(ds):
    """
    Pre-process the forest grass mask DataArray to remove noise values and re-align anomalies with 0, 1 and np.nan
    """
    ds = xr.where((ds < 0) | (ds > 1), np.nan, ds)
    return ds

def regrid_xr(xr_temp, xr_new, method = "nearest"):
    """
    Match up the dim/coord of two Dataset or DataArray objects using the interp function.
    """
    if ("latitude" in xr_temp.coords) and ("latitude" in xr_new.coords):
        return xr_new.interp(latitude = xr_temp.latitude, longitude = xr_temp.longitude, method=method)
    elif ("lat" in xr_temp.coords) and ("lat" in xr_new.coords):     
        return xr_new.interp(lat = xr_temp.lat, lon = xr_temp.lon, method=method)

if __name__=='__main__':
        
    dates_ = pd.date_range(datetime(2003,4,1), datetime(2020,6,30), freq='D')
    
    #Load shapefile for clipping:
    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA/PID90109_VIC_Boundary_SHP_FWA.shp")
#    shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp")
    areas_list = shp_in['Area_Name'].values
#    areas_list = ['Mallee', 'Wimmera', 'Northern Country', 'North East', 'South West', 'Central', 'North Central']
    avg_curing = {area_name+'_curing': [] for area_name in areas_list}
    
    #Load grass forest mask TIF and process:
    grass_forest_mask_tif_path = 'C://Users/clark/analysis1/afdrs_fbi_recalc/data/fuel/v4_forest0grass1malleeheath1heathland0.tif'
    grass_forest_mask = loadGeoTiff(grass_forest_mask_tif_path, da_var_name='grass_forest', as_Dataset = True)
    grass_forest_mask = preprocess_forest_grass_mask(grass_forest_mask)
    # Re-grid mask to VicClim data
    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/pre2017_curing_for_VicClim_200001.nc")
    grass_forest_mask = regrid_xr(curing_in, grass_forest_mask, method = "nearest")
    curing_in.close()
    
    mask_switch_check=0
    
    for yr in dates_.year.unique():
        print("Commencing "+str(yr))
        months_in_range = dates_[dates_.year==yr].month.unique()
        for mth in range(months_in_range.min(), months_in_range.max()+1):
                #Load input weather files:
                print('Loading '+str(mth))
                if mth<10:
                    mth_str = '0'+str(mth)
                else:
                    mth_str = str(mth)
                    
                if (yr<=2016 | ((yr==2017) & (mth<=6))):
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/pre2017_curing_for_VicClim_"+str(yr)+mth_str+".nc")
                else:
                    curing_in = xr.open_dataset("M:/Archived/VicClimV5/Curing_Input_for_VicClim/mapVictoria_curing_for_VicClim_"+str(yr)+mth_str+".nc")
                    if mask_switch_check==0:
                        grass_forest_mask = grass_forest_mask.rename({'lat': 'latitude', 'lon': 'longitude'})
                        mask_switch_check=1
                
                #Filter by grass forest mask:
                curing_in = xr.where(grass_forest_mask['grass_forest']==1, curing_in['GCI'], np.nan)
                
                #clip to each region sequentially and get average
                for area_name in areas_list:
                    area_polygon = shp_in[shp_in['Area_Name']==area_name]
                    #Get clip of temp and RH according to area:
                    if mask_switch_check==0:
                        curing_in.rio.set_spatial_dims(x_dim='lon',y_dim='lat',inplace=True) 
                    else:
                        curing_in.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) 
                    curing_in.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
                    curing_clipped = curing_in.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
                
                    curing_avg = np.nanmean(curing_clipped, axis=(0,1))
                    avg_curing[area_name+'_curing'] = np.append(avg_curing[area_name+'_curing'], curing_avg)
                
                print(str(yr)+str(mth)+" done.")
                del(curing_avg)
                del(curing_clipped)
                curing_in.close()
    
    curing_avg_df = pd.DataFrame(avg_curing)
    curing_avg_df.index = dates_
    curing_avg_df.to_csv('./incidents_fmc_data/vicclim_avg_curing_'+str(dates_[0].year)+"0"+str(dates_[0].month)+"-"+str(dates_[-1].year)+str(dates_[-1].month)+".csv")
    
                    