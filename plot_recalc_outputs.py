#Plots the outputs to the recalculated fire danger grids, compare to official outputs

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geopandas

#Set dates:
year_sel_ = 2023
mon_sel = 12
day_sel = 28

if mon_sel < 10:
    mon_sel_str = '0'+str(mon_sel)
else:
    mon_sel_str = str(mon_sel)
    
if day_sel < 10:
    day_sel_str = '0'+str(day_sel)
else:
    day_sel_str = str(day_sel)
#load the file:
#recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/mallee_cases_completed_v1/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
recalc_file_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/VIC_"+str(year_sel_)+mon_sel_str+day_sel_str+"_recalc.nc")
official_fdi_in = xr.open_dataset("M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/2023-2024/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str+"/IDV71116_VIC_MaxFDI_SFC.nc.gz")
official_fbi_in = xr.open_dataset("M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/2023-2024/"+str(year_sel_)+"_"+mon_sel_str+"_"+day_sel_str+"/IDZ10137_AUS_AFDRS_max_fbi_SFC.nc.gz")
"""
Find the maximum FBI and FDI at each point: 
Note - there is a need to grab the correct time window, midnight to midnight lOCAL time.    
"""
start_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+day_sel_str+'T13:00:00')
end_time_ = np.datetime64(str(year_sel_)+'-'+mon_sel_str+'-'+str(day_sel+1)+'T12:00:00')
start_ind = np.where(recalc_file_in.time.values==start_time_)[0][0]
end_ind = np.where(recalc_file_in.time.values==end_time_)[0][0]
mask = official_fbi_in.land_sea_mask
max_recalc_fdi = recalc_file_in['FDI_SFC'][:,:,start_ind:end_ind].max(dim='time', keep_attrs=True)
max_recalc_fbi = recalc_file_in['index_1'][start_ind:end_ind,:,:].max(dim='time', keep_attrs=True)

"""Load fire weather area (FWA) shapefile for plotting"""
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")

"""
Plot:
Note at this stage we are plotting day +1 in the official BOM FDI/FBI as the forecast is this
"""

fig, axs = plt.subplots(2,2,figsize=(10,7), subplot_kw={'projection': ccrs.PlateCarree()})

max_recalc_fdi.plot(ax=axs[0,0], transform=ccrs.PlateCarree(),vmin=0, vmax=80., cmap='viridis')
shp_in.plot(ax=axs[0,0], facecolor="none")
max_recalc_fbi.plot(ax=axs[0,1], transform=ccrs.PlateCarree(),vmin=0., vmax=80., cmap='viridis')
shp_in.plot(ax=axs[0,1], facecolor="none")
official_fdi_in['MaxFDI_SFC'][:,:,1].where(mask).plot(ax=axs[1,0], transform=ccrs.PlateCarree(), vmin=0., vmax=80., cmap='viridis')
shp_in.plot(ax=axs[1,0], facecolor="none")
official_fbi_in['MaxFBI_SFC'][1,:,:].plot(ax=axs[1,1], transform=ccrs.PlateCarree(), vmin=0., vmax=80., cmap='viridis')
shp_in.plot(ax=axs[1,1], facecolor="none")
print(recalc_file_in['time'][22].values)
print(official_fbi_in['time'][1].values)
axs[0,0].coastlines()
axs[0,0].set_title('Recalc FDI')
axs[0,0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
axs[0,0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
axs[0,0].set_extent([140.8,145,-37.8,-33.8])
axs[0,1].coastlines()
axs[0,1].set_title('Recalc FBI')
axs[0,1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
axs[0,1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
axs[0,1].set_extent([140.8,145,-37.8,-33.8])
axs[1,0].coastlines()
axs[1,0].set_title('FDI from BOM forecast')
axs[1,0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
axs[1,0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
axs[1,0].set_extent([140.8,145,-37.8,-33.8])
axs[1,1].coastlines()
axs[1,1].set_title('FBI from BoM forecast')
axs[1,1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
axs[1,1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
axs[1,1].set_extent([140.8,145,-37.8,-33.8])
plt.savefig("fdi_fbi_recalc_vs_bom"+str(year_sel_)+mon_sel_str+str(day_sel+1)+".png")

"""Calc difference and plot"""
difference_fdi = max_recalc_fdi - official_fdi_in['MaxFDI_SFC'][:,:,1]
difference_fbi = max_recalc_fbi - official_fbi_in['MaxFBI_SFC'][1,:,:]

fig2, axs2 = plt.subplots(1,2,figsize=(9,5), subplot_kw={'projection': ccrs.PlateCarree()})

difference_fdi.plot(ax=axs2[0], transform=ccrs.PlateCarree(),vmin=-20., vmax=20., cmap='RdYlGn_r')
shp_in.plot(ax=axs2[0], facecolor="none")
difference_fbi.plot(ax=axs2[1], transform=ccrs.PlateCarree(),vmin=-20., vmax=20., cmap='RdYlGn_r')
shp_in.plot(ax=axs2[1], facecolor="none")
axs2[0].coastlines()
axs2[0].set_title('FDI difference recalc - BOM')
axs2[0].gridlines(draw_labels=False)
axs2[0].set_extent([140.8,145,-37.8,-33.8])
axs2[1].coastlines()
axs2[1].set_title('FBI difference recalc - BOM')
axs2[1].gridlines(draw_labels=False)
axs2[1].set_extent([140.8,145,-37.8,-33.8])

plt.savefig("fdi_fbi_recalc_vs_bom_"+str(year_sel_)+mon_sel_str+str(day_sel+1)+"difference.png")

recalc_file_in.close()
official_fdi_in.close()
official_fbi_in.close()