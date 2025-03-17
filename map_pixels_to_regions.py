"""
For each of the pixels: Which region does it lie in? 
Assumes the pixel co-ordinates are at the centroid of the pixel
"""

import numpy as np
import xarray as xr
import geopandas
import pandas as pd

import shapely.geometry
from shapely.geometry import Point

#Load a netcdf grid:
#This one for a 3km grid:
grids_in = xr.open_dataset('C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/cases_control/VIC_20250103_recalc.nc')
#This one for a 1.5 km grid:
#grids_in = xr.open_dataset('C:/Users/clark/analysis1/afdrs_fbi_recalc/Recalculated_VIC_Grids/full_recalc_jul_24/recalc_files/VIC_20171002_recalc.nc')
    
#Load a shapefile:
#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc-main/data/shp/PID90409_VIC_Boundary_SHP_LGA/PID90409_VIC_Boundary_SHP_LGA.shp", crs='ESPG:4326')
#shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/DEECA_fire_districts.shp", crs='ESPG:4326')
shp_in = geopandas.read_file("C://Users/clark/analysis1/afdrs_fbi_recalc/data/shp/ICC/ICC.shp", crs='ESPG:4326')

#Build coordinate list:
lat_list_ = grids_in.latitude.values
lon_list_ = grids_in.longitude.values
coord_list = [(a, b) for a in lat_list_ for b in lon_list_]
lat_list2 = [a[0] for a in coord_list]
lon_list2 = [a[1] for a in coord_list]
gdf_points = pd.DataFrame({'longitude': lon_list2, 'latitude':lat_list2})
gdf_points['coords'] = list(zip(gdf_points['longitude'],gdf_points['latitude']))
gdf_points['coords'] = gdf_points['coords'].apply(Point)

#Do the join:
points_ = geopandas.GeoDataFrame(gdf_points, geometry='coords', crs=shp_in.crs)
pointInPolys = geopandas.tools.sjoin(points_, shp_in, predicate="within", how='left')
pointInPolys = pointInPolys.set_index(['latitude','longitude'])
#For ICCs at 1.5km resolution, despite the odds we get a point EXACTLY on a boundary.
#So we need to remove the duplicate. Just keep the first one - it turns out we assign
#it to Alexandra, not Shepparton. This is likely to make minimal difference - just boost
#the sampling for Alexandra ever so slightly because it's smaller.
pointInPolys = pointInPolys[~pointInPolys.index.duplicated(keep='first')]
points_xarr = xr.Dataset.from_dataframe(pointInPolys)

"""
#For DEECA districts:rename to use the same structure as the other templates
points_xarr = points_xarr.drop_vars('REGION_NAM')
points_xarr = points_xarr.rename_vars({'OBJECTID': 'Area_ID', 'DISTRICT_N': 'Area_Name'})
"""

#For new ICC districts, rename also:
points_xarr = points_xarr.drop_vars(['Shape_Leng', 'Shape_Area'])
points_xarr = points_xarr.rename_vars({'ICC_NAME':'Area_Name'})

#Finalise and save:
points_xarr=points_xarr.drop_vars(['coords','index_right'])
points_xarr.to_netcdf("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_ICC_3km.nc")

grids_in.close()
