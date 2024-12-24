import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import matplotlib as mpl
import geopandas
from datetime import datetime, timedelta
import seaborn

def plot_and_compare_fbi(max_fbi1, max_fbi2, areas_shapefile, extent, save_plot=None):
    """
    Plot two FBI cases and a comparison plot from xarray DataArrays with lat and lon
    coordinate variables. Ie. three panels.

    Parameters
    ----------
    max_fbi1 (xarray DataArray) : FBI array 1, must be 2-dimensional with lat and lon coordinate variables.
    max_fbi2 (xarray DataArray): FBI array 2 that we want to compare.
    areas_shapefile (geopandas object): Shapefile that contains sub-regions that will be plotted as outlines overlaying
    the FBI data.
    save_plot (string, optional) : Name to save the plot. 

    Returns
    -------
    No output arguments. Saves plot as a PNG with the name given by save_plot.

    """
    
    fig, axs = plt.subplots(1,3,figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap_rating = pltcolors.ListedColormap(['white','green','gold','darkorange','darkred'])
    #    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5], cmap_rating.N)
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,200], cmap_rating.N)

    im1 = max_fbi1.plot(ax=axs[0], transform=ccrs.PlateCarree(),cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.035)
#    cb1.set_label(label='FBI',size=14)
    cb1.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    im2 = max_fbi2.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical', fraction=0.035)
#    cb2.set_label(label='Maximum FBI in day',size=14)
    cb2.ax.tick_params(labelsize=14)
    
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI 1')
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
    axs[0].set_extent(extent)
    axs[1].coastlines()
    axs[1].set_title('FBI 2')
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[1].set_extent(extent)
#    axs[1].set_extent([140.8,144.7,-37.3,-34.2])

    """Calc difference and plot"""
    difference_fbi = max_fbi2 - max_fbi1
    
    im3 = difference_fbi.plot(ax=axs[2], transform=ccrs.PlateCarree(),vmin=-20., vmax=20., cmap='RdYlGn_r', add_colorbar=False)
    cb3 = plt.colorbar(im3, orientation='vertical', fraction=0.035)
    cb3.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[2], facecolor="none")
    axs[2].coastlines()
    axs[2].set_title('Diff 2 - 1')
    axs[2].gridlines(draw_labels=False)
#    axs[2].set_extent([140.8,144.7,-37.0,-33.8])
    axs[2].set_extent(extent)
    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')

    
def plot_fbi_and_rating_with_fwas(FBI,rating,areas_shapefile, box_extent=None, save_plot=None):
    """
    Plots FBI and fire danger rating from xarray DataArray inputs. 

    Parameters
    ----------
    FBI : FBI array, must be 2-dimensional with lat and lon coordinate variables.
    rating : Fire danger rating associated with the set FBI above.
    areas_shapefile (geopandas object): Shapefile that contains sub-regions that will be plotted as outlines overlaying
    the FBI data.
    save_plot (string, optional) : Name to save the plot. 

    Returns
    -------
    None. Saves plot as a PNG with the name given by save_plot.

    """
    if box_extent is None:
        box_extent = [140.8,150,-39.3,-33.8]
    
    fig, axs = plt.subplots(1,2,figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cmap_rating = pltcolors.ListedColormap(['blue','green','gold','darkorange','red'])
    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5], cmap_rating.N)
    """
    cmap_rating = pltcolors.ListedColormap(['blue','green'])
    norm = pltcolors.BoundaryNorm([0,1,2], cmap_rating.N)
    """
    im1 = FBI.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=50., cmap='viridis', add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical')
    cb1.set_label(label='FBI',size=14)
    cb1.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    im2 = rating.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical')
    cb2.set_label(label='Rating',size=14)
    cb2.ax.tick_params(labelsize=14)
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI', fontsize=16)
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,147.8,-38.4,-33.8])
    axs[0].set_extent(box_extent)
#    axs[0].set_extent([143,146.2,-37.3,-34.5])  #Northern Country
    axs[1].coastlines()
    axs[1].set_title('Rating',fontsize=16)
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    #axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[1].set_extent([140.8,147.8,-38.4,-33.8])
    axs[1].set_extent(box_extent)
#    axs[1].set_extent([143,146.2,-37.3,-34.5])         #Northern Country

    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def plot_index_comparison_histogram(index1, index2, index_name_1=None, index_name_2=None, title_head=None, out_file_path=None):
    """
    Plots a histogram for two fire index arrays, in order to compare them.

    Parameters
    ----------
    index1 : Array containing either FBI or FDI.
    
    index2 : Same form as index1, the data we want to compare it to.
    
    index_name_1 (optional): Label name for index1
    
    index_name_2 (optional): Label name for index2
    
    title_head (optional): Header title of plot.

    out_file_path (optional): File name (and/or path) to save the histogram.

    Returns
    -------
    Histogram of the values of index1 and index2.

    """
    
    colors_ = seaborn.color_palette('bright')
    if index_name_1 is None:
        index_name_1='index_1'
    if index_name_2 is None:
        index_name_2 = 'index_2'
    
    seaborn.histplot(index1, label=index_name_1, bins=np.arange(1,50,2), color=colors_[0])
    seaborn.histplot(index2, label=index_name_2, bins=np.arange(1,50,2), color=colors_[1])
    
    plt.legend()
    
    if title_head is not None:
        plt.title(title_head)
    
    if out_file_path is not None:
        plt.savefig(out_file_path)

def scatterplot_fbi_vs_fdi_dominants(fdi, fbi, dominant_model_strs, title_str=None, out_file_path=None):
    fig, ax = plt.subplots(figsize=(11,8))
    seaborn.set(font_scale=2.2)
    seaborn.scatterplot(x=fdi, y=fbi, hue=dominant_model_strs, s=36, palette={'Grassland': 'royalblue', 'Vesta': 'limegreen', 'Mallee heath': 'darkorange', 'None dominant': 'k'})
    ax.axhline(y=0, color='k',linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_xlabel("McArthur FDI", fontsize=26)
    ax.set_ylabel("AFDRS FBI", fontsize=26)
    #plot also an x=y line:
    seaborn.lineplot(x=np.linspace(0,100,5), y=np.linspace(0,100,5), color='k')

    if title_str is not None:
        plt.title(title_str, fontsize=32)

    if out_file_path is not None:
        plt.savefig(out_file_path)

def scatterplot_fbi_vs_fdi_months(fdi, fbi, month_list, title_str=None, out_file_path=None):
    fig, ax = plt.subplots(figsize=(11,8))
    seaborn.set(font_scale=2.2)
    seaborn.scatterplot(x=fdi, y=fbi, hue=month_list, s=36, palette={9: 'lime', 10: 'limegreen', 11: 'forestgreen', 
                                                                     12: 'gold', 1: 'orange', 2: 'darkorange',
                                                                     3: 'fuchsia', 4: 'mediumorchid', 5: 'rebeccapurple',
                                                                     6: 'royalblue', 7: 'blue', 8: 'navy'})
    ax.axhline(y=0, color='k',linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_xlabel("McArthur FDI", fontsize=22)
    ax.set_ylabel("AFDRS FBI", fontsize=22)
    #plot also an x=y line:
    seaborn.lineplot(x=np.linspace(0,100,5), y=np.linspace(0,100,5), color='k')

    if title_str is not None:
        plt.title(title_str)

    if out_file_path is not None:
        plt.savefig(out_file_path)

def scatterplot_fbi_vs_fdi_seasons(fdi, fbi, season_list, title_str=None, out_file_path=None):
    fig, ax = plt.subplots(figsize=(11,8))
    seaborn.set(font_scale=2.2)
    seaborn.scatterplot(x=fdi, y=fbi, hue=season_list, s=36, palette={4: 'limegreen', 1: 'orange', 2: 'fuchsia', 3:'blue'})
    ax.axhline(y=0, color='k',linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_xlabel("McArthur FDI", fontsize=22)
    ax.set_ylabel("AFDRS FBI", fontsize=22)
    #plot also an x=y line:
    seaborn.lineplot(x=np.linspace(0,100,5), y=np.linspace(0,100,5), color='k')
    plt.legend([ "Sep-Nov","Dec-Feb", "Mar-May", "Jun-Aug"])

    if title_str is not None:
        plt.title(title_str)

    if out_file_path is not None:
        plt.savefig(out_file_path)

def plot_df(df, areas_shapefile, extent, save_plot=None):
    """
    Plot drought factor on a single plot.

    Parameters
    ----------
    df (xarray DataArray) : Drought factor array, must be 2-dimensional with lat and lon coordinate variables.
    areas_shapefile (geopandas object): Shapefile that contains sub-regions that will be plotted as outlines overlaying
    the FBI data.
    save_plot (string, optional) : Name to save the plot. 

    Returns
    -------
    No output arguments. Option to save plot as a PNG with the name given by save_plot.

    """
    
    fig, axs = plt.subplots(1,figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap_df = pltcolors.ListedColormap(['blue','royalblue','darkturquoise','aquamarine','springgreen','gold','darkorange','red','darkred'])
#    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5,6,7,8,9,10,10.1], cmap_df.N)
    norm = pltcolors.BoundaryNorm([3,4,5,6,7,8,9,9.5,10], cmap_df.N, extend='max')
    
    im1 = df.plot(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_df, norm=norm, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.03)
    cb1.set_label('Drought factor', size=16)
    cb1.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs, facecolor="none")

    axs.coastlines()
    axs.set_title('Drought factor 16 Nov 2024', fontsize=20)
    axs.set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs.set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
#    axs.set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
#    axs.set_extent([142.8,145.7,-39,-36.3]) #Melbourne and a little west and north... 
#    axs.set_extent([147.0,150,-38.1,-36.4])   #East Gippsland
    axs.set_extent(extent)   #East Gippsland
    axs.set_xlabel('')
    axs.set_ylabel('')
    
    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def plot_kbdi(kbdi, areas_shapefile, extent, save_plot=None):
    """
    Plot KBDI on a single plot.

    Parameters
    ----------
    df (xarray DataArray) : KBDI array, must be 2-dimensional with lat and lon coordinate variables.
    areas_shapefile (geopandas object): Shapefile that contains sub-regions that will be plotted as outlines overlaying
    the FBI data.
    save_plot (string, optional) : Name to save the plot. 

    Returns
    -------
    No output arguments. Option to save plot as a PNG with the name given by save_plot.

    """
    
    fig, axs = plt.subplots(1,figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
    cmap_df = pltcolors.ListedColormap(['blue','royalblue','darkturquoise','aquamarine','springgreen','gold','darkorange','red','darkred'])
#    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5,6,7,8,9,10,10.1], cmap_df.N)
    norm = pltcolors.BoundaryNorm([10,20,30,40,50,75,100,150,200], cmap_df.N, extend='max')
    
    im1 = kbdi.plot(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_df, norm=norm, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.03)
    cb1.set_label('KBDI', size=16)
    cb1.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs, facecolor="none")

    axs.coastlines()
    axs.set_title('KBDI', fontsize=20)
    axs.set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs.set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
#    axs.set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
#    axs.set_extent([142.8,145.7,-39,-36.3]) #Melbourne and a little west and north... 
#    axs.set_extent([147.0,150,-38.1,-36.4])   #East Gippsland
    axs.set_extent(extent)   #East Gippsland
    axs.set_xlabel('')
    axs.set_ylabel('')

    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def plot_fbi_and_fdi(fbi, fdi, areas_shapefile, extent, save_plot=None):
    fig, axs = plt.subplots(1,2,figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})

    cmap_rating = pltcolors.ListedColormap(['white','green','gold','darkorange','darkred'])
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,200], cmap_rating.N)
    im1= fbi.plot(ax=axs[0], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
#    im1= fbi.plot(ax=axs[0], transform=ccrs.PlateCarree(), cmap='viridis',add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.04)
    cb1.set_label(label='FBI',size=16)
    cb1.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    im2 = fdi.plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=0., vmax=100., cmap='viridis', add_colorbar=False)
#    im2 = fdi.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical', fraction=0.04)
    cb2.set_label(label='FDI',size=16)
    cb2.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI', size=18)
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[0].set_extent(extent)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('')
    axs[1].coastlines()
    axs[1].set_title('FDI', size=18)
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[1].set_extent(extent) 
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')

    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def plot_fbi_ffdi_gfdi_ratings(fbi, ffdi, gfdi, areas_shapefile, extent, save_plot=None):
    fig, axs = plt.subplots(3,2,figsize=(13,18), subplot_kw={'projection': ccrs.PlateCarree()})

    im1 = fbi.plot(ax=axs[0,0], transform=ccrs.PlateCarree(),vmin=0., vmax=100., cmap='viridis', add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical')
    cb1.set_label(label='FBI',size=18)
    cb1.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[0,0], facecolor="none")
    axs[0,0].coastlines()
    axs[0,0].set_title('FBI', fontsize=22)
    axs[0,0].set_xticks([], crs=ccrs.PlateCarree())
    axs[0,0].set_xlabel('')
    axs[0,0].set_yticks([], crs=ccrs.PlateCarree())
    axs[0,0].set_extent(extent)
    
    im2 = ffdi.plot(ax=axs[1,0], transform=ccrs.PlateCarree(), vmin=0., vmax=100., cmap='viridis', add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical')
    cb2.set_label(label='FBI',size=18)
    cb2.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[1,0], facecolor="none")
    axs[1,0].coastlines()
    axs[1,0].set_title('FFDI', fontsize=22)
    axs[1,0].set_xticks([], crs=ccrs.PlateCarree())
    axs[1,0].set_xlabel('')
    axs[1,0].set_yticks([], crs=ccrs.PlateCarree())
    axs[1,0].set_extent(extent)

    im3 = gfdi.plot(ax=axs[2,0], transform=ccrs.PlateCarree(), vmin=0., vmax=100., cmap='viridis', add_colorbar = False)
    cb3 = plt.colorbar(im3, orientation='vertical')
    cb3.set_label(label='FBI',size=18)
    cb3.ax.tick_params(labelsize=18)    
    areas_shapefile.plot(ax=axs[2,0], facecolor="none")
    axs[2,0].coastlines()
    axs[2,0].set_title('GFDI', fontsize=22)
    axs[2,0].set_xticks([], crs=ccrs.PlateCarree())
    axs[2,0].set_xlabel('')
    axs[2,0].set_yticks([], crs=ccrs.PlateCarree())
    axs[2,0].set_extent(extent)
    
    cmap_rating_fbi = pltcolors.ListedColormap(['white','green','gold','darkorange','darkred'])
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,200], cmap_rating_fbi.N)

    im4 = fbi.plot(ax=axs[0,1], transform=ccrs.PlateCarree(),cmap=cmap_rating_fbi, norm=norm, add_colorbar=False)
    cb4 = plt.colorbar(im4, orientation='vertical')
    cb4.set_label(label='FBI Rating',size=18)
    cb4.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[0,1], facecolor="none")
    
    cmap_rating_fdi = pltcolors.ListedColormap(['green','blue','gold','darkorange','red', 'darkred'])
    norm = pltcolors.BoundaryNorm([0,12,24,50,75,100,200], cmap_rating_fdi.N)
    im5 = ffdi.plot(ax=axs[1,1], transform=ccrs.PlateCarree(), cmap=cmap_rating_fdi, norm=norm, add_colorbar=False)
    cb5 = plt.colorbar(im5, orientation='vertical')
    cb5.set_label(label='FBI Rating',size=18)
    cb5.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[1,1], facecolor="none")
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,150,200], cmap_rating_fdi.N)
    im6= gfdi.plot(ax=axs[2,1], transform=ccrs.PlateCarree(), cmap=cmap_rating_fdi, norm=norm, add_colorbar=False)
    cb6 = plt.colorbar(im6, orientation='vertical')
    cb6.set_label(label='FFDI Rating',size=18)
    cb6.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[2,1], facecolor="none")
    axs[0,1].coastlines()
    axs[0,1].set_title('FBI Rating', fontsize=22)
    axs[0,1].set_xticks([], crs=ccrs.PlateCarree())
    axs[0,1].set_yticks([], crs=ccrs.PlateCarree())
    axs[0,1].set_extent(extent)
    axs[1,1].coastlines()
    axs[1,1].set_title('FFDI rating', fontsize=22)
    axs[1,1].set_xticks([], crs=ccrs.PlateCarree())
    axs[1,1].set_yticks([], crs=ccrs.PlateCarree())
    axs[1,1].set_extent(extent)

    axs[2,1].coastlines()
    axs[2,1].set_title('GFDI rating', fontsize=22)
    axs[2,1].set_xticks([], crs=ccrs.PlateCarree())
    axs[2,1].set_yticks([], crs=ccrs.PlateCarree())
    axs[2,1].set_extent(extent)   #Wimmera + SW

    axs[0,0].set_ylabel('')
    axs[1,0].set_ylabel('')
    axs[2,0].set_ylabel('')
    axs[0,1].set_ylabel('')
    axs[1,1].set_ylabel('')
    axs[2,1].set_ylabel('')
    axs[0,1].set_xlabel('')
    axs[1,1].set_xlabel('')
    axs[2,1].set_xlabel('')

    if save_plot is not None:
        plt.savefig(save_plot+'.png')
        
def plot_fbi_gfdi_cheney_ratings(fbi, gfdi, cheney_fdi, areas_shapefile, extent, save_plot=None):
    fig, axs = plt.subplots(1,3,figsize=(18,10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cmap_rating_fbi = pltcolors.ListedColormap(['white','green','gold','darkorange','darkred'])
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,200], cmap_rating_fbi.N)

    im4 = fbi.plot(ax=axs[0], transform=ccrs.PlateCarree(),cmap=cmap_rating_fbi, norm=norm, add_colorbar=False)
    cb4 = plt.colorbar(im4, orientation='vertical')
    cb4.set_label(label='FBI Rating',size=18)
    cb4.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    
    cmap_rating_fdi = pltcolors.ListedColormap(['green','blue','gold','darkorange','red', 'darkred'])
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,150,200], cmap_rating_fdi.N)
    im5 = gfdi.plot(ax=axs[1], transform=ccrs.PlateCarree(), cmap=cmap_rating_fdi, norm=norm, add_colorbar=False)
    cb5 = plt.colorbar(im5, orientation='vertical')
    cb5.set_label(label='GFDI Rating',size=18)
    cb5.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    cmap_rating_cheney = pltcolors.ListedColormap(['white','green','gold','darkorange','darkred'])
    norm = pltcolors.BoundaryNorm([0,12,24,50,100,200], cmap_rating_fdi.N)
    im6= cheney_fdi.plot(ax=axs[2], transform=ccrs.PlateCarree(), cmap=cmap_rating_cheney, norm=norm, add_colorbar=False)
    cb6 = plt.colorbar(im6, orientation='vertical')
    cb6.set_label(label='Cheney Rating',size=18)
    cb6.ax.tick_params(labelsize=18)
    areas_shapefile.plot(ax=axs[2], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI Rating', fontsize=22)
    axs[0].set_xticks([], crs=ccrs.PlateCarree())
    axs[0].set_yticks([], crs=ccrs.PlateCarree())
    axs[0].set_extent(extent)
    axs[1].coastlines()
    axs[1].set_title('GFDI rating', fontsize=22)
    axs[1].set_xticks([], crs=ccrs.PlateCarree())
    axs[1].set_yticks([], crs=ccrs.PlateCarree())
    axs[1].set_extent(extent)

    axs[2].coastlines()
    axs[2].set_title('Cheney rating', fontsize=22)
    axs[2].set_xticks([], crs=ccrs.PlateCarree())
    axs[2].set_yticks([], crs=ccrs.PlateCarree())
    axs[2].set_extent(extent)

    axs[0].set_ylabel('')
    axs[1].set_ylabel('')
    axs[2].set_ylabel('')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[2].set_xlabel('')

    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def plot_varpanel(temp, rh, wind, df, areas_shapefile, extent, save_plot=None):
    fig, axs = plt.subplots(2,2,figsize=(11,8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    cmap_temp = mpl.cm.jet
    norm = pltcolors.BoundaryNorm([10,15,20,25,30,35,40,45], cmap_temp.N)
    im1 = temp.plot(ax=axs[0,0], transform=ccrs.PlateCarree(), cmap=cmap_temp, norm=norm, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical', fraction=0.036)
    cb1.set_label('Temp', size=16)
    cb1.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs[0,0], facecolor="none")
    axs[0,0].coastlines()
    axs[0,0].set_title('Max temp', fontsize=20)
    axs[0,0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0,0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[0,0].set_extent(extent)   
    axs[0,0].set_xlabel('')
    axs[0,0].set_ylabel('')

    cmap_rh = mpl.cm.gist_earth_r
    norm = pltcolors.BoundaryNorm([5,10,15,20,30,40,50,60], cmap_rh.N)
    im2 = rh.plot(ax=axs[0,1], transform=ccrs.PlateCarree(), cmap=cmap_rh, norm=norm, add_colorbar=False)
    cb2 = plt.colorbar(im2, orientation='vertical', fraction=0.036)
    cb2.set_label('RH', size=16)
    cb2.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs[0,1], facecolor="none")
    axs[0,1].coastlines()
    axs[0,1].set_title('Min RH', fontsize=20)
    axs[0,1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0,1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[0,1].set_extent(extent)   #Wimmera + SW
    axs[0,1].set_xlabel('')
    axs[0,1].set_ylabel('')


    cmap_wind= mpl.cm.BuPu
    norm = pltcolors.BoundaryNorm([10,20,30,40,50,60], cmap_wind.N)
    im3 = wind.plot(ax=axs[1,0], transform=ccrs.PlateCarree(), cmap=cmap_wind, norm=norm, add_colorbar=False)
    cb3 = plt.colorbar(im3, orientation='vertical', fraction=0.036)
    cb3.set_label('Wind (km/h)', size=16)
    cb3.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs[1,0], facecolor="none")
    axs[1,0].coastlines()
    axs[1,0].set_title('Wind', fontsize=20)
    axs[1,0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1,0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[1,0].set_extent(extent)   #Wimmera + SW
    axs[1,0].set_xlabel('')
    axs[1,0].set_ylabel('')
    
    cmap_df = pltcolors.ListedColormap(['blue','royalblue','darkturquoise','aquamarine','springgreen','gold','darkorange','red','darkred'])
    norm = pltcolors.BoundaryNorm([3,4,5,6,7,8,9,9.5,10], cmap_df.N, extend='max')
    im4 = df.plot(ax=axs[1,1], transform=ccrs.PlateCarree(), cmap=cmap_df, norm=norm, add_colorbar=False)
    cb4 = plt.colorbar(im4, orientation='vertical', fraction=0.036)
    cb4.set_label('', size=16)
    cb4.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs[1,1], facecolor="none")
    axs[1,1].coastlines()
    axs[1,1].set_title('Drought factor', fontsize=20)
    axs[1,1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1,1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[1,1].set_extent(extent)   #Wimmera + SW
    axs[1,1].set_xlabel('')
    axs[1,1].set_ylabel('')
    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')
        

def plot_curing(curing, areas_shapefile, save_plot=None):
    """
    Plot grass curing on a single plot.

    Parameters
    ----------
    curing (xarray DataArray) : Drought factor array, must be 2-dimensional with lat and lon coordinate variables.
    areas_shapefile (geopandas object): Shapefile that contains sub-regions that will be plotted as outlines overlaying
    the FBI data.
    save_plot (string, optional) : Name to save the plot. 

    Returns
    -------
    No output arguments. Option to save plot as a PNG with the name given by save_plot.

    """
    
    fig, axs = plt.subplots(1,figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
#    cmap_c = pltcolors.ListedColormap(['mediumblue','blue','royalblue','cornflowerblue','darkturquoise','aquamarine','springgreen','gold','darkorange','red','darkred'])
    cmap_c = plt.get_cmap('nipy_spectral')
    cmap_tr= pltcolors.LinearSegmentedColormap.from_list('nipy_spectral_trunc_0.45_0.9', cmap_c(np.linspace(0.40,0.9,100)))
#    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5,6,7,8,9,10,10.1], cmap_c.N)
    
#    im1 = curing.plot(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=False)
    im1 = curing.plot(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_tr, vmin=0, vmax=100, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical')
    cb1.set_label('Curing %', size=16)
    cb1.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs, facecolor="none")

    axs.coastlines()
    axs.set_title('Curing used in FBI', fontsize=20)
    axs.set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs.set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
    axs.set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
 
    axs.set_xlabel('')
    axs.set_ylabel('')
    
    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')


def plot_fmc(fmc_in, areas_shapefile, extent, save_plot=None):
    """
    Plot two FBI cases and a comparison plot from xarray DataArrays with lat and lon
    coordinate variables. Ie. three panels.

    Parameters
    ----------
    max_fbi1 (xarray DataArray) : FBI array 1, must be 2-dimensional with lat and lon coordinate variables.
    max_fbi2 (xarray DataArray): FBI array 2 that we want to compare.
    areas_shapefile (geopandas object): Shapefile that contains sub-regions that will be plotted as outlines overlaying
    the FBI data.
    save_plot (string, optional) : Name to save the plot. 

    Returns
    -------
    No output arguments. Saves plot as a PNG with the name given by save_plot.

    """
    
    fig, axs = plt.subplots(1,1,figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})

    fmc_in.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=0.3, cmap='viridis')
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    axs.coastlines()
    axs.set_title('FBI 1')
    axs.set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs.set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
    axs.set_extent(extent)
    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def fbi_rating_calc_spatial(fbi):
    # setup rating array of same size as FBI
    rating = np.full(fbi.shape, np.nan)

    # assign ratings based on FBI thresholds
    rating[fbi >= 100] = 4 # Catastrophic
    rating[fbi < 100] = 3 # Extreme
    rating[fbi < 50] = 2 # High
    rating[fbi < 24] = 1 # Moderate
    rating[fbi < 12] = 0 # No rating
    return rating

def ffdi_rating_calc_spatial(ffdi):
    # setup rating array of same size as FBI
    rating = np.full(ffdi.shape, np.nan)

    # assign ratings based on FBI thresholds
    rating[ffdi >= 100] = 6 # Catastrophic
    rating[ffdi < 100] = 5 # Extreme
    rating[ffdi < 75] = 4 # Severe
    rating[ffdi < 50] = 3 # Very High
    rating[ffdi < 24] = 2 # High
    rating[ffdi < 12] = 1 # Low-Moderate
    return rating

def gfdi_rating_calc_spatial(gfdi):
    # setup rating array of same size as FBI
    rating = np.full(gfdi.shape, np.nan)

    # assign ratings based on FBI thresholds
    rating[gfdi >= 150] = 6 # Catastrophic
    rating[gfdi < 150] = 5 # Extreme
    rating[gfdi < 100] = 4 # Severe
    rating[gfdi < 50] = 3 # Very High
    rating[gfdi < 24] = 2 # High
    rating[gfdi < 12] = 1 # Low-Moderate
    return rating