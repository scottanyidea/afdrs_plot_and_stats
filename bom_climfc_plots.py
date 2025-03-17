#Play around and plot BoM seasonal forecast data.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import cartopy.crs as ccrs
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
#load data:
path_to_folder = "C://Users/clark/analysis1/Bom_clim_fc_data/"
median_fn_data = xr.open_dataset(path_to_folder+"IDCK000103.index_1.forecast.raw.median.aus.fortnightly.20250208.nc")
median_wk_data = xr.open_dataset(path_to_folder+"IDCK000103.index_1.forecast.raw.median.aus.weekly.20250126.nc")
median_mth_data = xr.open_dataset(path_to_folder+"IDCK000103.index_1.forecast.raw.median.aus.monthly.20250123.nc")


#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/PID90109_VIC_Boundary_SHP_FWA\PID90109_VIC_Boundary_SHP_FWA.shp")
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/ICC/ICC.shp")

median_fn_pr = median_fn_data['percentage_of_ensembles'][1,0,:,:]

#plot:
fig, axs = plt.subplots(1, figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
extent = [140.8,150.2,-39.3,-33.8] #all Vic
cmap_c = plt.get_cmap('BrBG_r')
cmap_tr= pltcolors.LinearSegmentedColormap.from_list('BrBGr_trunc_0.2_0.8', cmap_c(np.linspace(0.1,0.9,100)))
im1 = median_fn_pr.plot(ax=axs, transform=ccrs.PlateCarree(), vmin=20, vmax=80, cmap=cmap_tr, add_colorbar=False)
#im1 = median_mth_data['percentage_of_ensembles'][1,0,:,:].plot(ax=axs, transform=ccrs.PlateCarree(), vmin=20, vmax=80, cmap=cmap_tr, add_colorbar=False)
cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.035,  extend='both')
cb1.set_label('% Chance of exceeding', size=14)
cb1.ax.tick_params(labelsize=13)
axs.set_title('Chance of exceeding median AFDRS index - 8-21 Feb 2025', fontsize=18)
axs.coastlines()
shp_in.plot(ax=axs, facecolor="None")
axs.set_extent(extent)
plt.savefig('vic_chance_exceed_fbi_8-21feb')

#region_list =np.unique(shp_in['Area_Name'])
region_list =np.unique(shp_in['ICC_NAME'])
region_avg_prob = []
median_fn_pr.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude',inplace=True) #OK doing this makes a lot more sense now there's a time dimension. It's telling rioxarray what the spatial dims are!
median_fn_pr.rio.write_crs("EPSG:4326",inplace=True)  #And now tell it the coord reference system.
for region in region_list:
#    area_polygon = shp_in[shp_in['Area_Name']==region]
    area_polygon = shp_in[shp_in['ICC_NAME']==region]
    clipped_data_ = median_fn_pr.rio.clip(area_polygon.geometry.apply(mapping), area_polygon.crs, drop=False)
    region_avg_prob.append(np.nanmean(clipped_data_))


region_cur_df = pd.DataFrame(data={'ICC Footprint':region_list, 'Avg pr of exceeding median':region_avg_prob})
region_cur_df.to_csv('ICC_footprint_avg_pr_of_exceeding_medianFBI_8-21Feb.csv')
