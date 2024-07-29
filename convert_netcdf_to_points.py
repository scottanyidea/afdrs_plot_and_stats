
"""
This code takes a netcdf file from FBI and converts it into points so we can properly apply
the algorithm to clip points by geometries
"""

# Standard
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import numpy as np
import gzip
import shutil
import urllib.request
from contextlib import closing
import h5py, h5netcdf, xarray as xr
import warnings
import io
import glob

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap

from affine import Affine
import rasterio
import geopandas as gpd
from shapely.geometry import Point

from geopandas.tools import sjoin

#Load the file:
template_from_recalcs = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jan_24/recalc_files\VIC_20171002_recalc.nc")
template_from_recalcs = template_from_recalcs.isel(time=[0]).drop("time").squeeze('time')

lat_len=len(list(template_from_recalcs.latitude.values))
lon_len=len(list(template_from_recalcs.longitude.values))
na_unique_id = np.arange(0, lat_len * lon_len).reshape(lat_len,lon_len)
template_from_recalcs['PIXEL_REF_ID'] = (('latitude', 'longitude'), na_unique_id)

da_nan = np.full((lat_len, lon_len), np.nan)
da_nan = np.expand_dims(da_nan, axis=0)
ds_vic = xr.Dataset()
ds_vic.coords['longitude'] = ('longitude', template_from_recalcs.longitude.values)
ds_vic.coords['latitude'] = ('latitude', template_from_recalcs.latitude.values)
ds_vic.coords['time'] = ('time', pd.date_range('1970-01-01','1970-01-01', name='time'))

ds_vic['VIC_GRID'] = (('time','latitude','longitude'),da_nan)
ds_vic['PIXEL_REF_ID'] = template_from_recalcs['PIXEL_REF_ID']
ds_vic.coords['latitude'] = ds_vic.coords['latitude'].assign_attrs(units='degrees_N')
ds_vic.coords['longitude'] = ds_vic.coords['longitude'].assign_attrs(units='degrees_E')
ds_vic.attrs['Conventions'] = "COARDS"
ds_vic.attrs['creationTime'] = int(time.time())
ds_vic.attrs['creationTimeString'] = datetime.utcnow().isoformat()
ds_vic.attrs['generatedBy'] = "CFA"

ds_vic.to_netcdf('VIC_Boundary_Grid_ALL.nc', engine ="h5netcdf", encoding = {"VIC_GRID": {"compression": "gzip", "compression_opts": 5}})