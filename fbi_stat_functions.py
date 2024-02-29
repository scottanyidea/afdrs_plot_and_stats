# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:12:40 2024

@author: clark
"""
import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd
from datetime import datetime
import time
import sys, os

from pathlib import Path

def find_dominant_fuel_type_for_a_rating(fbi_arr, rating_val, fuel_type_map, fuel_lut_path):
    #This assumes the fbi_array is already clipped to the desired geometry and is 2D (ie. already maximum)
    #Fuel type map does NOT need to be clipped.
    
    #Mask fuel type map to be same as the FBI map:
    fuel_type_map_clipped = xr.where(fbi_arr, fuel_type_map, np.nan)
    fuel_type_map_clipped.name = 'fuel_type'
    
    #Merge FBI with fuel types, and mask to only those pixels above 90th percentile (or whatever we set rating_val to)
    merged_fbi_ft = xr.merge([fbi_arr, fuel_type_map_clipped])
    merged_fbi_ft = merged_fbi_ft.where((merged_fbi_ft['index_1'] >= rating_val))    #get only those values above say 90th percentile
    top_pixels_table = merged_fbi_ft.to_dataframe()
    top_pixels_table.dropna(axis=0, inplace=True)

    #Load the fuel lut to match fuel types to the codes and pixels:
    fuel_lut = pd.read_csv(fuel_lut_path)
    fuel_FBM_dict = pd.Series(fuel_lut.FBM.values,index=fuel_lut.FTno_State).to_dict()
    top_pixels_table['FBM'] = top_pixels_table['fuel_type'].map(fuel_FBM_dict)
    
    #If the highest ranked fuel model has less than half the points, return "none" as
    #we don't consider it dominant. Else, return the name of the model.
    #OR: if we have a small region, sometimes all the pixels have a zero FBI and it 
    #somehow messes up the grater than or equal to function even if threshold is also zero.
    #In these cases also set "None".
    if (len(top_pixels_table)==0) or (top_pixels_table.FBM.value_counts().iloc[0]/top_pixels_table.FBM.value_counts().sum() < 0.5):
        topmodel = 'None dominant'
    else:
        topmodel = top_pixels_table.FBM.value_counts().index[0]
        
    return topmodel  #ie. return the NAME of the top fuel type

def find_dominant_fuel_code_for_a_rating(fbi_arr, rating_val, fuel_type_map, fuel_lut_path, return_table=False):
    #This assumes the fbi_array is already clipped to the desired geometry and is 2D (ie. already maximum)
    #Fuel type map does NOT need to be clipped.
    
    #Mask fuel type map to be same as the FBI map:
    fuel_type_map_clipped = xr.where(fbi_arr, fuel_type_map, np.nan)
    fuel_type_map_clipped.name = 'fuel_type'
    
    #Merge FBI with fuel types, and mask to only those pixels above 90th percentile (or whatever we set rating_val to)
    merged_fbi_ft = xr.merge([fbi_arr, fuel_type_map_clipped])
    merged_fbi_ft = merged_fbi_ft.where((merged_fbi_ft['index_1'] >= rating_val))    #get only those values above say 90th percentile
    top_pixels_table = merged_fbi_ft.to_dataframe()
    top_pixels_table.dropna(axis=0, inplace=True)

    #Get table of fuel type pixel counts:
    count_table = top_pixels_table.fuel_type.value_counts()
    
    #If the highest ranked fuel model has less than half the points, return "none" as
    #we don't consider it dominant. Else, return the name of the model.
    #OR: if we have a small region, sometimes all the pixels have a zero FBI and it 
    #somehow messes up the grater than or equal to function even if threshold is also zero.
    #In these cases also set "None".
    if (len(top_pixels_table)==0) or (count_table.iloc[0]/count_table.sum() < 0.5):
        topcode = 'None dominant'
    else:
        topcode = count_table.index[0]
    
    if return_table==False:
        return topcode  #ie. return the NAME of the top fuel type
    else:
        return count_table/count_table.sum()  #ie. return as fractions of the total pixels
    
