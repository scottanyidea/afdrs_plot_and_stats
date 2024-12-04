#Replace KBDI and DF (or any others we want) within a
#region of choice, for input into fdrs_calcs.

import numpy as np
import xarray as xr
import geopandas
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


if __name__=="__main__":
    #load file
    file_path = 'M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/'
    
    year = 2024
    month_ = 11
    day_ = 19
    
    the_date_object = datetime(year=year, month=month_, day=day_)
    
    if month_<7:
        file_in = xr.open_dataset(file_path+str(year-1)+"-"+str(year)+"/"+str(year)+"_"+str(month_)+"_"+str(day_)+"/IDV71147_VIC_KBDI_SFC.nc.gz")
        file_in_df = xr.open_dataset(file_path+str(year-1)+"-"+str(year)+"/"+str(year)+"_"+str(month_)+"_"+str(day_)+"/IDV71127_VIC_DF_SFC.nc.gz")
    else:
        file_in = xr.open_dataset(file_path+str(year)+"-"+str(year+1)+"/"+str(year)+"_"+str(month_)+"_"+str(day_)+"/IDV71147_VIC_KBDI_SFC.nc.gz")
        file_in_df = xr.open_dataset(file_path+str(year)+"-"+str(year+1)+"/"+str(year)+"_"+str(month_)+"_"+str(day_)+"/IDV71127_VIC_DF_SFC.nc.gz")
        
    
    #Get LGA boundaries:
    region_template = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_LGA_3km.nc")
    
    #Modify data: 
    #Change KBDI
    kbdi_to_add = 40
    regions_to_change = ['Glenelg', 'Surf Coast', 'Colac-Otway', 'Greater Bendigo', 'Golden Plains']
    kbdi_out = file_in['KBDI_SFC']+xr.where(region_template['Area_Name'].isin(regions_to_change), kbdi_to_add, 0)

    #Save:
    kbdi_out_ds = kbdi_out.to_dataset(name='KBDI_SFC')
    kbdi_out_ds.to_netcdf("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/alternative_vars/KBDI_increased_SWlocs_"+str(kbdi_to_add)+"_"+str(year)+str(month_)+str(day_)+".nc")
    
    #The changed KBDI has an effect on DF. Assume that DF is at its upper limit when modified.
    #First need to merge datasets to align time steps.
    kbdi_df_merged = xr.merge([kbdi_out_ds, file_in_df])
    kbdi_df_merged['KBDI_SFC'] = kbdi_df_merged['KBDI_SFC'].ffill(dim='time')
    x_lim = (75/(270.525-1.267*kbdi_df_merged['KBDI_SFC']))   #this is equation for KBDI > 20. We're only applying this to added area for now but needs revision
    df_lim = 10.5*(1-np.exp(-(kbdi_df_merged['KBDI_SFC']+30)/40))*(41*np.power(x_lim, 2)+x_lim)/(40*np.power(x_lim,2)+x_lim+1)
    df_lim = xr.where(df_lim<=10, df_lim, 10)
    df_out = xr.where(region_template['Area_Name'].isin(regions_to_change), df_lim, kbdi_df_merged['DF_SFC'])    
    
    #Save:
    #Note - most likely the KBDI timestamp is way earlier than the first DF one because KBDI sets to the start of the day.
    #So all we need to do is remove the first blank DF time step.
    df_out = df_out[:,:,1:]
    df_out_ds = df_out.to_dataset(name='DF_SFC').transpose("time","latitude","longitude")  #why does it rearrange during this script???
    df_out_ds.to_netcdf("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/alternative_vars/DF_increased_SWlocs_"+str(kbdi_to_add)+"_"+str(year)+str(month_)+str(day_)+".nc")
    
    #Plot to be sure:
    fig, axs = plt.subplots(1,2, figsize=(14,8), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [140.8,150.2,-39,-33.8] #all Vic
    im1 = kbdi_out[1,:,:].plot(ax=axs[0], transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.051)
    im2 = df_out[:,:,6].plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap='viridis', add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical', fraction=0.051)