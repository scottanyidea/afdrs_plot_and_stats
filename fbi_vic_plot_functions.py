import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import geopandas
from datetime import datetime, timedelta
import seaborn

def plot_and_compare_fbi(max_fbi1, max_fbi2, areas_shapefile, save_plot=None):
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

    max_fbi1.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=50., cmap='viridis')
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    max_fbi2.plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=0., vmax=50., cmap='viridis')
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI 1')
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
    axs[0].set_extent([142.8,145.7,-38.8,-37.0])
    axs[1].coastlines()
    axs[1].set_title('FBI 2')
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    axs[1].set_extent([142.8,145.7,-38.8,-37.0])
#    axs[1].set_extent([140.8,144.7,-37.3,-34.2])

    """Calc difference and plot"""
    difference_fbi = max_fbi1 - max_fbi2
    
    difference_fbi.plot(ax=axs[2], transform=ccrs.PlateCarree(),vmin=-10., vmax=10., cmap='RdYlGn_r')
    areas_shapefile.plot(ax=axs[2], facecolor="none")
    axs[2].coastlines()
    axs[2].set_title('Diff recalc - BOM')
    axs[2].gridlines(draw_labels=False)
#    axs[2].set_extent([140.8,144.7,-37.0,-33.8])
    axs[2].set_extent([142.8,145.7,-38.8,-37.0])
    
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
    ax.set_xlabel("McArthur FDI", fontsize=22)
    ax.set_ylabel("AFDRS FBI", fontsize=22)
    #plot also an x=y line:
    seaborn.lineplot(x=np.linspace(0,100,5), y=np.linspace(0,100,5), color='k')

    if title_str is not None:
        plt.title(title_str)

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

def plot_df(df, areas_shapefile, save_plot=None):
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
    cmap_df = pltcolors.ListedColormap(['mediumblue','blue','royalblue','cornflowerblue','darkturquoise','aquamarine','springgreen','gold','darkorange','red','darkred'])
    norm = pltcolors.BoundaryNorm([0,1,2,3,4,5,6,7,8,9,10], cmap_df.N)
    
    im1 = df.plot(ax=axs, transform=ccrs.PlateCarree(), cmap=cmap_df, norm=norm, add_colorbar=False)
    cb1 = plt.colorbar(im1, orientation='vertical')
    cb1.set_label('Drought factor', size=16)
    cb1.ax.tick_params(labelsize=16)
    areas_shapefile.plot(ax=axs, facecolor="none")

    axs.coastlines()
    axs.set_title('Drought factor', fontsize=20)
    axs.set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs.set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
    axs.set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
 
    axs.set_xlabel('')
    axs.set_ylabel('')
    
    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')

def plot_fbi_and_fdi(fbi, fdi, areas_shapefile, save_plot=None):
    fig, axs = plt.subplots(1,2,figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})

    fbi.plot(ax=axs[0], transform=ccrs.PlateCarree(),vmin=0., vmax=100., cmap='viridis')
    areas_shapefile.plot(ax=axs[0], facecolor="none")
    fdi.plot(ax=axs[1], transform=ccrs.PlateCarree(), vmin=0., vmax=100., cmap='viridis')
    areas_shapefile.plot(ax=axs[1], facecolor="none")
    axs[0].coastlines()
    axs[0].set_title('FBI')
    axs[0].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[0].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
#    axs[0].set_extent([140.8,144.7,-37.3,-34.2])
#    axs[0].set_extent([142.8,145.7,-38.8,-37.0])
    axs[0].set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
    axs[1].coastlines()
    axs[1].set_title('FDI')
    axs[1].set_xticks([142,144,146,148,150], crs=ccrs.PlateCarree())
    axs[1].set_yticks([-38,-36,-34], crs=ccrs.PlateCarree())
    #axs[1].set_extent([142.8,145.7,-38.8,-37.0])
    axs[1].set_extent([140.8,145.7,-39,-33.8])   #Wimmera + SW
#    axs[1].set_extent([140.8,144.7,-37.3,-34.2])

    
    if save_plot is not None:
        plt.savefig(save_plot+'.png')
        


def fbi_rating_calc_spatial(fbi):
    if fbi < 12:
        rating = 0
    else:
        if fbi < 24:
            rating = 1
        else:
            if fbi < 50:
                rating = 2
            else:
                if fbi <100:
                    rating = 3
                else:
                    rating = 4
    return rating