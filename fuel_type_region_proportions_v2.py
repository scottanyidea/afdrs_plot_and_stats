#Count number of pixels in each region by fuel type
#This new version uses the template files built using the
#geopandas script - so we ensure the point centroids are actually
#within the LGA rather than using rioxarray naively

import numpy as np
import xarray as xr
import geopandas
import rioxarray
from shapely.geometry import mapping
import pandas as pd

#load an FBI recalc file. any one that has all fuel types in it.
fbi_data = xr.open_dataset('C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_jan_24/recalc_files/VIC_20180202_recalc.nc')
#fbi_data = xr.open_dataset('C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_control/VIC_20240124_recalc.nc')
fuel_type_map = fbi_data['fuel_type'][1,:,:]

#load template file:
template_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_PDD_1500m.nc")

#merge:
data_in_frame = xr.merge([fuel_type_map, template_in]).to_dataframe()
#Set up dict of fuel type codes:
#TODO: Pipe this into the fuel lut.
fuel_type_dict = {}

fuel_type_dict['Forest'] = [3005, 3006, 3007, 3008, 3009, 3010, 3014, 3018, 3019, 3022, 3027, 3028, 3033, 3035, 3036, 3039, 3040, 3041, 3043, 3045, 3061, 3063, 3065, 3066, 3090, 3091, 3092, 3099]
fuel_type_dict['Wet_forest'] = [3002,3011,3012,3013,3015,3032]
fuel_type_dict['Grass'] = [3004,3016,3020,3042,3044,3046,3062,3064]
fuel_type_dict['Pine'] = [3080,3081,3082,3098]
fuel_type_dict['Heath'] = [3001,3017,3021,3023,3024,3031,3048]
fuel_type_dict['Mallee_heath'] = [3025,3026,3049,3050,3051]
fuel_type_dict['Low_wetland'] = [3029, 3030, 3034, 3037]
fuel_type_dict['Chenopod_shrubland'] = [3003]
fuel_type_dict['Woody_horticulture'] = [3097]
fuel_type_dict['Urban'] = [3047, 3095, 3096]
fuel_type_dict['Non_combustible'] = [3000, 3038]
"""
fuel_type_dict['3066 Little Desert forest'] = [3066]
"""
#Set up pandas dataframe to save the values.
df_out = pd.DataFrame(list(fuel_type_dict.keys()), columns=['Fuel_type'])

data_in_frame = data_in_frame.dropna(subset=['Area_Id'])
region_list = data_in_frame['Area_Name'].unique()
region_list.sort()

for reg in region_list:
    #Get the fuel types in the region.
    ft_list_region = data_in_frame[data_in_frame['Area_Name']==reg]['fuel_type']
    
    #Count total pixels
    pixels_total = ft_list_region.count()
    
    #Count pixels in each fuel type
    no_pixels_in_type = []
    for ftyp in list(fuel_type_dict.keys()):
        npit = np.isin(ft_list_region, fuel_type_dict[ftyp])
        no_pixels_in_type.append(npit.sum())
    
    #Add column to dataframe
    df_out[reg] = no_pixels_in_type/pixels_total
    
#Save dataframe:
#df_out.to_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/fuel_type_frac_wimmera.csv", index=False)
df_out.to_csv("C:/Users/clark/OneDrive - Country Fire Authority/analysis_results/fuel_type_fracs_tables/fuel_type_frac_by_pdd_1500m_centroid_feb24.csv", index=False)