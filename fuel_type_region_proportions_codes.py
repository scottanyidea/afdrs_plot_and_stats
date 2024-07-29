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
fbi_data = xr.open_dataset('C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/full_recalc_mar_24/recalc_files/VIC_20180102_recalc.nc')
#fbi_data = xr.open_dataset('C:/Users/clark/analysis1/afdrs_fbi_recalc-main/Recalculated_VIC_Grids/cases_control/VIC_20240124_recalc.nc')
fuel_type_map = fbi_data['fuel_type'][1,:,:]

#load template file:
template_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_FWA_1500m.nc")
#template_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/template_nc_grids/map_by_pixel_centroid_LGA_1500m.nc")

#merge:
data_in_frame = xr.merge([fuel_type_map, template_in]).to_dataframe()

#Get list of fuel type codes from fuel lut:
fuel_lut = pd.read_csv("C:/Users/clark/analysis1/afdrs_fbi_recalc-main/data/fuel/fuel-type-model-authorised-vic-20231214033606.csv")
fuel_codes = fuel_lut['FTno_State'].sort_values()
    
#Set up pandas dataframe to save the values.
df_out = pd.DataFrame(list(fuel_codes), columns=['Fuel_type'])
region_list = data_in_frame['Area_Name'].unique()
region_list.sort()

data_in_frame = data_in_frame.dropna(subset=['Area_Id'])

for reg in region_list:
    #Get the fuel types in the region.
    ft_list_region = data_in_frame[data_in_frame['Area_Name']==reg]['fuel_type']

    #Count total pixels
    pixels_total = ft_list_region.count()
    
    #Count pixels in each fuel type:
    no_pixels_in_type = []
    for ftyp in fuel_codes:
        npit = np.isin(ft_list_region, ftyp)
        no_pixels_in_type.append(npit.sum())
    
    #Add to dataframe:
    df_out[reg] = no_pixels_in_type/pixels_total


df_out.to_csv("C:/Users/clark/OneDrive - Country Fire Authority/analysis_results/fuel_type_fracs_tables/fuel_code_frac_by_fwa_1500m_centroid_feb24.csv", index=False)
template_in.close()
fbi_data.close()