#Replace grass curing in region of choice to a new value to
#investigate impacts

import numpy as np
import xarray as xr
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import cartopy.crs as ccrs


if __name__=="__main__":
    #load file
    file_path = 'M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/'
    
    year = 2024
    month_ = 11
    day_ = 16
    
    the_date_object = datetime(year=year, month=month_, day=day_)
    
    if month_<7:
        file_in = xr.open_dataset(file_path+str(year-1)+"-"+str(year)+"/"+str(year)+"_"+str(month_)+"_"+str(day_)+"/IDZ10148_AUS_FSE_curing_SFC.nc.gz")
    else:
        file_in = xr.open_dataset(file_path+str(year)+"-"+str(year+1)+"/"+str(year)+"_"+str(month_)+"_"+str(day_)+"//IDZ10148_AUS_FSE_curing_SFC.nc.gz")
        
    
    #Get LGA boundaries:
    #region_template = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_LGA_3km.nc")
    region_template = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_FWA_3km.nc")
    
    #Modify data: 
    #Use a "floor" approach. Pixels that are below the minimum 
    curing_set = 90
    regions_to_change = ['Wimmera', 'South West', 'Central', 'Northern Country', 'North Central']
    curing_out = file_in['grass_curing']
    curing_out = xr.where((region_template['Area_Name'].isin(regions_to_change) & (curing_out<curing_set)), curing_set, curing_out)

    #Save:
    curing_out_ds = curing_out.to_dataset(name='grass_curing').transpose('time','latitude','longitude')
    curing_out_ds.to_netcdf("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/alternative_vars/curing_increased_westernthird_"+str(curing_set)+"_"+str(year)+str(month_)+str(day_)+".nc")
    
    #Plot to be sure:
    fig, axs = plt.subplots(1, figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [140.8,150.2,-39,-33.8] #all Vic
    cmap_c = plt.get_cmap('nipy_spectral')
    cmap_tr= pltcolors.LinearSegmentedColormap.from_list('nipy_spectral_trunc_0.45_0.9', cmap_c(np.linspace(0.40,0.9,100)))
    im1 = curing_out[:,:,1].plot(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_tr, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.051)
    