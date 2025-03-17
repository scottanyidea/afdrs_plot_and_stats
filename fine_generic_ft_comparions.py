#Comparison of fine and generic fuel types from AWS data.

import numpy as np
import pandas as pd

if __name__=='__main__':
    
    fbi_data_in = pd.read_csv('C:/Users/clark/analysis1/compiled_obs/AWS_fine_fuels/obs_202324_fine_fbis_max_genericcalc_3007.csv')
    
    #Use number of days above 24 (High rating) as a metric.
    day_count1 = fbi_data_in.groupby('station_full').apply(lambda df: sum(df['primary FBI'] >= 24))
    day_count2 = fbi_data_in.groupby('station_full').apply(lambda df: sum(df['primary_fine_FBI'] >= 24))
    primary_generic = fbi_data_in.groupby('station_full').agg({'primary FBM': 'first'})
    primary_fine = fbi_data_in.groupby('station_full').agg({'primary_fine_fuel_type_code': 'first'})
    
    day_count3 = fbi_data_in.groupby('station_full').apply(lambda df: sum(df['secondary FBI'] >= 24))
    day_count4 = fbi_data_in.groupby('station_full').apply(lambda df: sum(df['secondary_fine_FBI'] >= 24))
    secondary_generic = fbi_data_in.groupby('station_full').agg({'secondary FBM': 'first'})
    secondary_fine = fbi_data_in.groupby('station_full').agg({'secondary_fine_fuel_type_code': 'first'})
    
    day_count5 = fbi_data_in.groupby('station_full').apply(lambda df: sum(df['Generic_forest_FBI_recalc'] >=24))
    
    comparison_df = pd.concat([primary_generic, day_count1, primary_fine, day_count2], axis=1)
    comparison_df = comparison_df.rename(columns={0: 'primary >=24', 1:'primary fine >=24'})
    comparison_df['difference_prim'] = comparison_df['primary fine >=24']-comparison_df['primary >=24']

#    comparison_df.to_csv('C:/Users/clark/analysis1/compiled_obs/primary_days_above_24_FBI.csv')    
    primary_comparison_highdays = comparison_df[(comparison_df['difference_prim'] > 1) | (comparison_df['difference_prim'] < -1)]
    
    comparison_df_sec = pd.concat([secondary_generic, day_count3, secondary_fine, day_count4], axis=1)
    comparison_df_sec = comparison_df_sec.rename(columns={0: 'secondary >=24', 1:'secondary fine >=24'})
    comparison_df_sec['difference_sec'] = comparison_df_sec['secondary fine >=24']-comparison_df_sec['secondary >=24']
    secondary_comparison_highdays = comparison_df_sec[(comparison_df_sec['difference_sec'] > 1) | (comparison_df_sec['difference_sec'] < -1)]
#    comparison_df_sec.to_csv('C:/Users/clark/analysis1/compiled_obs/secondary_days_above_24_FBI.csv')

    comparison_df_comb = pd.concat([comparison_df, comparison_df_sec], axis=1)
    #comparison_df_comb.to_csv('C:/Users/clark/analysis1/compiled_obs/combined_days_above_24_FBI_3020_3007.csv')

    #This is a very inefficient way to calculate the difference and remove all the columns, but other 
    #ways just mess up weirdly...
    """
    comparison_df_gen_recalc = pd.concat([primary_generic, day_count1, secondary_generic, day_count3, day_count5], axis=1)
    comparison_df_gen_recalc = comparison_df_gen_recalc.rename(columns={0: 'primary >=24', 1:'secondary >=24', 2:'generic forest recalc >=24'})
    diff_1 = comparison_df_gen_recalc['generic forest recalc >=24'].values - comparison_df_gen_recalc['primary >=24'].values
    diff_2 = comparison_df_gen_recalc['generic forest recalc >=24'].values - comparison_df_gen_recalc['secondary >=24'].values
    diff_2a = np.where(comparison_df_gen_recalc['secondary FBM']=='Forest', diff_2, np.nan)
    diff_a = np.where(comparison_df_gen_recalc['primary FBM']=='Forest', diff_1, diff_2a)
    comparison_df_gen_recalc['difference_gen'] = diff_a
    
    comparison_df_gen_recalc['original forest >=24'] = np.where(comparison_df_gen_recalc['primary FBM']=='Forest', comparison_df_gen_recalc['primary >=24'].values, comparison_df_gen_recalc['secondary >=24'].values,)
    comparison_df_gen_recalc = comparison_df_gen_recalc.drop(columns=['primary FBM', 'primary >=24', 'secondary FBM', 'secondary >=24'])
    comparison_df_gen_recalc = comparison_df_gen_recalc[['original forest >=24', 'generic forest recalc >=24', 'difference_gen']]
    
    comparison_df_gen_recalc.to_csv('C://Users/clark/analysis1/compiled_obs/combined_days_above_24_FBI_genericrecalc.csv')
    """  
    comparison_df_gen_recalc = pd.concat([primary_fine, day_count2, secondary_fine, day_count4, day_count5], axis=1)
    comparison_df_gen_recalc = comparison_df_gen_recalc.rename(columns={0: 'primary_fine >=24', 1:'secondary_fine >=24', 2:'generic forest recalc >=24'})
    diff_1 = -(comparison_df_gen_recalc['generic forest recalc >=24'].values - comparison_df_gen_recalc['primary_fine >=24'].values)
    diff_2 = -(comparison_df_gen_recalc['generic forest recalc >=24'].values - comparison_df_gen_recalc['secondary_fine >=24'].values)
    diff_2a = np.where(comparison_df_gen_recalc['secondary_fine_fuel_type_code']==3007, diff_2, np.nan)
    diff_a = np.where(comparison_df_gen_recalc['primary_fine_fuel_type_code']==3007, diff_1, diff_2a)
    comparison_df_gen_recalc['difference_gen'] = diff_a
    
    comparison_df_gen_recalc['new forest 3007 >=24'] = np.where(comparison_df_gen_recalc['primary_fine_fuel_type_code']==3007, comparison_df_gen_recalc['primary_fine >=24'].values, comparison_df_gen_recalc['secondary_fine >=24'].values,)
    comparison_df_gen_recalc = comparison_df_gen_recalc.drop(columns=['primary_fine_fuel_type_code', 'primary_fine >=24', 'secondary_fine_fuel_type_code', 'secondary_fine >=24'])
    comparison_df_gen_recalc = comparison_df_gen_recalc[['generic forest recalc >=24', 'new forest 3007 >=24', 'difference_gen']]
    
    comparison_df_gen_recalc.to_csv('C://Users/clark/analysis1/compiled_obs/AWS_fine_fuels/combined_days_above_24_FBI_genrecalc_3007.csv')
                                                  

    #print(primary_comparison_highdays)
    