#Retrieve FBI from archived forecasts

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path

if __name__=="__main__":
    
    dates_ = pd.date_range(datetime(2024,10,1), datetime(2025, 4,30), freq='D')
    
    arch_folder = 'M:/Archived/FUEL_BASED_FDI_PRODUCTS_BACKUP/'
    #file_var = "IDZ10137_AUS_AFDRS_max_fbi_SFC.nc.gz"
    file_var = "IDV71116_VIC_MaxFDI_SFC.nc.gz"
    
    #shp_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_FWA_3km.nc")
    shp_in = xr.open_dataset("C:/Users/clark/analysis1/afdrs_fbi_recalc/data/template_nc_grids/map_by_pixel_centroid_ICC_3km.nc")
    area_list = np.unique(shp_in['Area_Name'].values)  #get unique area names
    area_list = np.delete(area_list, np.where(area_list==''))  #remove the empty area
    
    fbi_out = []
    k = 0
    for dt in dates_:
        if dt.month >=7:
            arch_path = arch_folder+str(dt.year)+'-'+str(dt.year+1)+'/'+str(dt.year)+'_'+dt.strftime("%m")+'_'+dt.strftime("%d")
        else:            
            arch_path = arch_folder+str(dt.year-1)+'-'+str(dt.year)+'/'+str(dt.year)+'_'+dt.strftime("%m")+'_'+dt.strftime("%d")
        
        if Path(arch_path+'/'+file_var).is_file():
            print("Loading "+dt.strftime("%Y%m%d"))
            arch_fbi = xr.open_dataset(arch_path+"/"+file_var)
            dt_sel = dt
        else:
            print("Missing "+dt.strftime("%Y%m%d")+". Using previous file.")
            dt_sel = dt_sel+timedelta(days=1)
            k = k+1
            
#        fbi_daily = arch_fbi.sel(time=dt_sel)['MaxFBI_SFC']
        fbi_daily = arch_fbi.sel(time=dt_sel)['MaxFDI_SFC']
        fbi_daily_out = []
        for area_name in area_list:
            fbi_trim = xr.where(shp_in['Area_Name']==area_name, fbi_daily, np.nan)
            fbi_val = np.nanpercentile(fbi_trim, 90)
            fbi_daily_out.append(fbi_val)
        
        fbi_out.append(fbi_daily_out)
    
    fbi_out_df = pd.DataFrame(fbi_out).set_axis(area_list+"_FBI", axis=1)
    fbi_out_df.index = dates_
    fbi_out_df.to_csv("./outputs/fbi_from_fc/FDI_daily_fromfc_20241001_20250430_icc.csv")
    print("There were "+str(k)+" missing dates in the period chosen.")