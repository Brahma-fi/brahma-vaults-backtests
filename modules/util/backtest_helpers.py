from datetime import datetime
import pandas as pd


def get_str_date(date: datetime):
    """

    :param date:
    :return:
    """
    day = (int(not (date.day // 10)) * '0') + f'{date.day}'
    month = (int(not (date.month // 10)) * '0') + f'{date.month}'
    year = f'{date.year % 2000}'

    output = day + month + year

    return output

def load_csv_in_chunks(master_path: str, chunksize: int=10e4):

    with pd.read_csv(master_path, chunksize=chunksize) as reader:
        for chunk in reader:
             yield chunk


def get_data_from_master(start_date: datetime, end_date: datetime,
                         master_path: str=None, master_data: pd.DataFrame=None):
    if master_data is None:
        master_data = pd.concat([chunk for chunk in load_csv_in_chunks(master_path)])


    output =  master_data.loc[master_data.index.between(start_date, end_date)]

    return output

def ts_signal_prep(tr_scan_labels_short, tr_scan_labels_long, confidence=1):
    """

    :param tr_scan_labels_short:
    :param tr_scan_labels_long:
    :param confidence:
    :return:
    """

    long_signals = (tr_scan_labels_short['t_value'] > confidence) & (tr_scan_labels_long['t_value'] > confidence)

    short_signals = (tr_scan_labels_short['t_value'] <= -confidence) & (tr_scan_labels_long['t_value'] < -confidence)

    positions = pd.DataFrame(index=long_signals.index)
    positions['position'] = 0
    positions.loc[long_signals, 'position'] = 1
    positions.loc[short_signals, 'position'] = -1
    positions.index = pd.to_datetime(positions.index)

    return positions
