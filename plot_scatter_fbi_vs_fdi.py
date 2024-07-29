"""
Produce a scatter plot of FBI vs FDI for two outputs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from fbi_vic_plot_functions import scatterplot_fbi_vs_fdi_dominants, scatterplot_fbi_vs_fdi_seasons

path_to_data = "C:/Users/clark/analysis1/datacube_daily_stats/version_mar24_2/data_tables/"
#path_to_data = "C:/Users/clark/analysis1/datacube_daily_stats/version_feb24/"

region = "East Gippsland"

data_in = pd.read_csv(path_to_data+region+"_datacube_2017-2022_fbi_rating.csv")
fbi_vals = data_in['FBI']
fdi_vals = data_in['McArthur_FDI']
dom_models_vals = data_in['Dominant FT']

title_head = 'FBI vs FDI - '+region
out_file_path = "C:/Users/clark/analysis1/datacube_daily_stats/version_mar24_2/scatterplots/"+region+"_FBI_FDI_scatter.png"
scatterplot_fbi_vs_fdi_dominants(fdi_vals, fbi_vals, dom_models_vals, title_str=title_head, out_file_path=out_file_path)
#scatterplot_fbi_vs_fdi_dominants(fdi_vals, fbi_vals, dom_models_vals, title_str=title_head)

months_list = pd.to_datetime(data_in['Date'], format="%Y-%m-%d").dt.month
#Now I decided I want it by season.
seasons = [1,1,2,2,2,3,3,3,4,4,4,1]
month_to_season = dict(zip(range(1,13),seasons))
seasons_list = months_list.map(month_to_season)
out_file_path_m = "C:/Users/clark/analysis1/datacube_daily_stats/version_mar24_2/scatterplots/"+region+"_FBI_FDI_scatter_seasonss.png"
scatterplot_fbi_vs_fdi_seasons(fdi_vals, fbi_vals, seasons_list, title_str=title_head, out_file_path=out_file_path_m)
