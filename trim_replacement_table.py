# -*- coding: utf-8 -*-
"""
Take the replace fueltype all table and trim it to a form that is exactly the same as
the table produced in "datacube_region_designate_rating_all".

This is so you can simply cut and paste the table into the new set of statistics.
"""

import numpy as np
import pandas as pd

#Load original data:
original_in = pd.read_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_feb24/datacube_2017-2022_fbi_rating_cfa.csv")

#Load replacement data:
replacement_in = pd.read_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_mar24/fbi_changes_ltldes_mallee_cfa.csv")

#Get names of the areas
header_list = list(original_in.columns[2:])
location_names = np.unique([wd.split('_')[0] for wd in header_list])

for loc_name in location_names:
    print('Replacing '+loc_name)
    original_in[loc_name+'_FBI'] = replacement_in[loc_name+'_Changed_FBI']
    original_in[loc_name+'_Rating'] = replacement_in[loc_name+'_Changed_rating']
    original_in[loc_name+'_Dominant FT'] = replacement_in[loc_name+'_Changed_dominant FT']

original_in.to_csv("C:/Users/clark/analysis1/datacube_daily_stats/version_mar24/data_tables/datacube_2017-2022_fbi_rating_cfa.csv")