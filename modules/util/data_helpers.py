import pandas as pd

def group_historical_data(input_data, group_period, grouping_column):
    """

    :param input_data (pd.DataFrame): DataFrame of historical data indexed by datetime
    :param group_period (str): Timeperiod that historical data must be divided into eg 7d
    :param grouping_column (str): Column of input_data to use to group eg block_number or index - must be monotonic
    

    :return grouped_dates (pd.DataFrame): Dataframe describing grouped periods, integer index with start and end timestamps
    :return grouped_data (groupby obj): Pandas groupby object accessible via grouped_data.get_group(i)  
    """
    index_name = input_data.index.name
    
    group_values = input_data[grouping_column].resample(group_period,offset = '8h').last()
    mask = input_data[grouping_column].isin(group_values.values)
    partition_dates = pd.DataFrame(input_data[mask].index).drop_duplicates().reset_index().drop(columns='index')
    
    partition_dates['end_timestamp'] = partition_dates[index_name].shift(-1)
    
    # Drop last two rows to avoid stub periods. 
    partition_dates.drop(index = partition_dates.index[-2:],inplace=True)

    input_data['period'] = float("nan")

    for i,row in partition_dates.iterrows():
        period_mask = (input_data.index >= row[index_name]) & (input_data.index < row.end_timestamp)
        input_data.loc[period_mask,'period'] = i
    
    grouped_data = input_data.dropna().groupby('period')

    return partition_dates,grouped_data
