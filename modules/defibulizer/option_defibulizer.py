"""
eth-1w-put
"""


import datetime
from typing import Union, List
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.interpolate import interp1d
from modules.defibulizer.offchain_defibulizer import DeFibulizer
from modules.options.basic_option_helpers import get_mark_iv


class OptionDeFibulizer:
    """
    Class constructor
    """
    def __init__(self,
                 start_date: Union[datetime.datetime, str] = None,
                 end_date: Union[datetime.datetime, str] = None,
                 wanted_assets: list = ['ETH'],
                 estimated_min_maturity: float = 1,
                 estimated_max_maturity: float = 30):
        """
        Initializes the OptionDeFibuliuzer which loads the raw datasets
        from which the required option data will be extracted into the class memory.
        :param start_date: (datetime.datetime, str) Start date of the data sample
        :param end_date: (datetime.datetime, str) End date of the data sample
        :param wanted_assets: (list) Name of the underlying asset you want to get option trading data for.
                                     ex.: ['ETH', 'BTC', 'SOL']
        :param estimated_min_maturity: (flat) Estimated minimum maturity (in days) of option trades to fetch.
        :param estimated_max_maturity: (float) Estimated maximum maturity (in days) of option trades to fetch.
        """

        self.dz = DeFibulizer()
        self.start_date = start_date
        self.end_date = end_date
        self.allowed_assets = ['BTC', 'ETH', 'SOL', 'USDC']
        self.wanted_asset_check(wanted_assets)
        self.dz = DeFibulizer()
        self.est_min_maturity = estimated_min_maturity
        self.est_max_maturity = estimated_max_maturity
        self.raw_data = self.get_clean_raw_data(requested_assets=wanted_assets)

        sns.set(palette='deep')

    def wanted_asset_check(self, wanted_assets: list) -> list:
        """
        Checks if the asset name is in the allowed assets list.
        :param wanted_assets:
        """

        check = all(asset in self.allowed_assets for asset in wanted_assets)

        if not check:
            raise ValueError(f'Invalid asset data requested.'
                             f' Make sure the requested assets are among {self.allowed_assets}')

    def get_raw_data(self, requested_assets: Union[str, List[str]] = None) -> Union[dict, pd.DataFrame]:
        """
        Gets the raw Data from the DeFibulizer option data processing call.
        :param requested_assets: (str, list) Asset/s to fetch raw option trade data for
        :return:
        """

        if len(requested_assets) != 0:
            asset_names = []
            raw_asset_names = dict()

            for asset in requested_assets:
                asset_name = asset + '-OPT'
                asset_names.append(asset_name)
                raw_asset_names[asset] = {}

            temp_data = self.dz.get_data(asset_names=asset_names,
                                         start_date=self.start_date,
                                         end_date=self.end_date,
                                         joint_dataframe=False)

            raw_data = dict(zip(requested_assets, temp_data))

        else:
            raise TypeError('Incorrect asset option_type, please provide either str or List[str].')

        return raw_data

    def get_clean_raw_data(self, requested_assets: Union[str, List[str]] = None) -> Union[dict, pd.DataFrame]:
        """
        Cleans the raw option trade data. Removes surplus columns and filters based on expected minimum
        and maximum maturities required.
        :param requested_assets: (str, list) Asset/s to fetch option trade data for.
        :return: (pd.DataFrame)
        """
        raw_data = self.get_raw_data(requested_assets=requested_assets)

        expiry_timestamp = lambda date: datetime.datetime(date.year, date.month, date.day, 8,
                                                          tzinfo=datetime.timezone.utc).timestamp() * 1e3

        raw_data_cleaned = {}
        drop_columns = ['trade_seq', 'trade_id', 'tick_direction', 'block_trade_id', 'liquidation']
        for asset in requested_assets:
            temp_data = raw_data[asset].copy()
            temp_data.sort_index(inplace=True)

            temp_data.drop(columns=drop_columns, inplace=True)

            temp_data['expiry_timestamp'] = temp_data.expiry_date.apply(expiry_timestamp)

            temp_data.strike = pd.to_numeric(temp_data.strike)
            min_expiries_mask = (temp_data.expiry_timestamp - temp_data.timestamp) / 1e3 >= self.est_min_maturity*86400
            max_expiries_mask = (temp_data.expiry_timestamp - temp_data.timestamp) / 1e3 <= self.est_max_maturity*86400
            temp_clean_data = temp_data.loc[min_expiries_mask & max_expiries_mask].copy()
            raw_data_cleaned[asset] = temp_clean_data

        return raw_data_cleaned

    def get_option_data(self, asset: str,
                        date_time: datetime.datetime = None,
                        option_type: str = None,
                        maturity: int = None,
                        window: int = None,
                        strike: int = None):
        """
        Gets the specific option data from the raw dataframe based on the
        provided specifications
        :param date_time: (datetime.datetime) Datetime from which the lookback period will be counted from.
        :param asset: (str) Underlying asset name for which the data will be extracted.
        :param option_type: (str) Option type: put('P') pr call('C')
        :param maturity: (int) Time to maturity.
        :param window: (int) Window period to select the data from in minutes.
        :param strike: (int) The strike price of the desired option.
        :return: (pd.DataFrame)
        """

        if asset not in self.raw_data.keys():
            raise ValueError('Asset needs to be in wanted_asset when intialising the class')

        initial_data = self.raw_data[asset].copy()

        initial_data['ttm'] = round((initial_data.expiry_timestamp - initial_data.timestamp)/1000/3600/24, 0)

        initial_data['iv'] = initial_data['iv'] / 100

        requirements = [option_type, strike, maturity]
        columns = ['option_type', 'strike', 'ttm']

        selection = True

        for req, col in zip(requirements, columns):
            if req is not None:
                initial_data = initial_data.loc[initial_data[col] == req]

        if window is not None:
            # fetch trades on either side of date_time with window width = window
            end = (date_time + datetime.timedelta(minutes=window/2)).timestamp()*1e3
            start = (date_time - datetime.timedelta(minutes=window/2)).timestamp()*1e3
            selection = (start < initial_data.timestamp) & (initial_data.timestamp <= end)

        output = initial_data.loc[selection].copy()

        output['mark_iv'] = output.apply(self.calculate_iv, axis=1)

        return output

    @staticmethod
    def calculate_iv(row: pd.Series):
        """
        Determines the mark_iv for a particular row of trade data.
        :param row: (pd.Series) Single row of trade data from raw_data DF.
        :return: (float)
        """

        is_call = (row.option_type == 'C')
        tau = (row.expiry_timestamp-row.timestamp)/1e3/3600/24/365

        mark_iv = get_mark_iv(spot=row.index_price, strike=row.strike, tau=tau, sigma=row.iv,
                              option_price=row.price, mark_price=row.mark_price, is_call=is_call)[0]
        return mark_iv

    def get_iv_data(self, asset: str, date_time: datetime.datetime,
                    maturity: int = None, option_type: str = None,
                    window: int = 120):
        """
        Gets the Implied Volatility data from a specific time based on the
        provided specifications
        :param date_time: (datetime.datetime) Datetime which will be the midpoint of the data fetched.
        :param asset: (str) Underlying asset name for which the data will be extracted.
        :param maturity: (int) Time to maturity.
        :param option_type: (str) Option type: put('P') pr call('C')
        :param window: (int) Window period width to select the data from in minutes.
        :return: (pd.DataFrame)
        """
        iv_data_raw = self.get_option_data(date_time=date_time,
                                           asset=asset, option_type=option_type,
                                           maturity=maturity, window=window)

        if maturity is not None:

            iv_data = iv_data_raw.groupby('strike',
                                          as_index=False)[['mark_iv', 'iv']].mean()

        else:

            iv_data = iv_data_raw.groupby(['expiry_date', 'ttm', 'strike'],
                                          as_index=False)[['strike', 'mark_iv', 'iv']].mean()

        return iv_data

    def get_iv_surface(self, asset: str, date_time: datetime.datetime,
                       window: int = 120):
        """
        Gets the Implied Volatility Surface from a specific point in time.
        :param asset: (str) Underlying asset name for which the data will be extracted.
        :param date_time: (datetime.datetime) Datetime which will be the midpoint of the data fetched.
        :param window: (int) Window period width to select the data from in minutes.
        :return: (pd.DataFrame)
        """
        iv_data_raw = self.get_option_data(date_time=date_time,
                                           asset=asset, window=window)

        iv_surface = iv_data_raw.groupby(['expiry_date', 'expiry_timestamp', 'strike'],
                                         as_index=False)[['strike', 'mark_iv']].mean()

        iv_surface['tau'] = (iv_surface.expiry_timestamp/1e3-date_time.timestamp())/3600/24/365

        return iv_surface

    def plot_iv_surface(self, asset: str, date_time: datetime.datetime,
                        option_type: str = None, window: int = 120,
                        use_mark: bool = True, view_init: list = [35, 45]):
        """
        :param asset: (str) Underlying asset name for which the data will be extracted.
        :param date_time: (datetime.datetime) Datetime which will be the midpoint of the data fetched.
        :param option_type: (str) Option type: put('P') pr call('C')
        :param window: (int) Window period width to select the data from in minutes.
        :param use_mark: (bool) Whether to use the mark_iv or trade_iv when plotting
        :param view_init:
        :return:
        """

        data = self.get_iv_data(asset=asset, date_time=date_time,
                                maturity=None, option_type=option_type,
                                window=window)

        iv_col = 'mark_iv' if use_mark else 'iv'

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(data.ttm, data.strike, data[iv_col], cmap=plt.cm.coolwarm, linewidth=0.1)
        ax.view_init(view_init[0], view_init[1])

        return fig

    def plot_iv_skew(self, asset: str, date_time: datetime.datetime,
                     maturity: int, option_type: str = None,
                     window: int = 120, use_mark: bool = True):
        """
        :param asset: (str) Underlying asset name for which the data will be extracted.
        :param date_time: (datetime.datetime) Datetime which will be the midpoint of the data fetched.
        :param option_type: (str) Option type: put('P') pr call('C')
        :param maturity: (int) Time to maturity in days
        :param window: (int) Window period width to select the data from in minutes.
        :param use_mark: (bool) Whether to use the mark_iv or trade_iv when plotting
        :return:
        """

        iv_col = 'mark_iv' if use_mark else 'iv'

        data = self.get_iv_data(asset=asset, date_time=date_time,
                                maturity=maturity, option_type=option_type,
                                window=window)

        plot = data.plot.scatter(x='strike', y=iv_col, figsize=(8, 8), color='#902df7')

        return plot

    @staticmethod
    def strike_interpolation(skew, strike):
        """
        :param skew: (pd.DataFrame) IV surface object with data from 1 expiry date.
        :param strike: (float) Strike price of option
        :return: (float) Implied volatility of option
        """
        for col in ['strike', 'mark_iv']:
            if col not in skew.columns:
                raise ValueError(f'Skew does not have required column: {col}')

        x = skew.strike.values
        y = skew.mark_iv.values
        f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))

        iv = f(strike)

        return iv

    def time_interpolation(self, near_skew, far_skew, tau, strike):
        """
        :param near_skew: (pd.DataFrame) IV surface object with data from 1 expiry date.
        :param far_skew: (pd.DataFrame) IV surface object with data from 1 expiry date.
        :param tau: (float) Time to maturity (in years) of option
        :param strike: (float) Strike price of option
        :return: (float) Implied volatility of option
        """
        for col in ['tau', 'strike', 'mark_iv']:
            if col not in near_skew.columns:
                raise ValueError(f'near_skew does not have required column: {col}')
            if col not in far_skew.columns:
                raise ValueError(f'far_skew does not have required column: {col}')

        near_iv = self.strike_interpolation(near_skew, strike)
        far_iv = self.strike_interpolation(far_skew, strike)

        near_tau = near_skew.tau.iloc[0]
        far_tau = far_skew.tau.iloc[0]

        cum_variance_near = near_iv ** 2 * near_tau
        cum_variance_far = far_iv ** 2 * far_tau

        f = interp1d([near_tau, far_tau], [cum_variance_near, cum_variance_far])

        cum_variance = f(tau)
        iv = np.sqrt(cum_variance / tau)

        return iv

    def surface_interpolation(self, iv_surface, tau, strike):
        """
        :param iv_surface: (pd.DataFrame) IV surface object from .get_iv_surface method
        :param tau: (float) Time to maturity (in years) of option
        :param strike: (float) Strike price of option
        :return: (float) IV of option
        """
        for col in ['tau', 'strike', 'mark_iv']:
            if col not in iv_surface.columns:
                raise ValueError(f'IV Surface does not have required column: {col}')

        iv_surface.sort_values(['tau', 'strike'], inplace=True)
        unique_expiries = iv_surface.tau.unique()

        if tau in unique_expiries:
            skew = iv_surface[iv_surface.tau == tau][['strike', 'mark_iv']]
            iv = self.strike_interpolation(skew, strike) + 0

        elif tau < unique_expiries.min():
            skew = iv_surface[iv_surface.tau == unique_expiries.min()][['strike', 'mark_iv']]
            iv = self.strike_interpolation(skew, strike) + 0

        elif tau > unique_expiries.max():
            skew = iv_surface[iv_surface.tau == unique_expiries.max()][['strike', 'mark_iv']]
            iv = self.strike_interpolation(skew, strike) + 0

        else:
            less_than_mask = unique_expiries < tau
            near_expiry_idx = np.where(less_than_mask)[0][-1]

            greater_than_mask = unique_expiries > tau
            far_expiry_idx = np.where(greater_than_mask)[0][0]

            near_skew = iv_surface[iv_surface.tau == unique_expiries[near_expiry_idx]][['strike', 'tau', 'mark_iv']]
            far_skew = iv_surface[iv_surface.tau == unique_expiries[far_expiry_idx]][['strike', 'tau', 'mark_iv']]

            iv = self.time_interpolation(near_skew, far_skew, tau, strike)

        return iv


class HistoricalOptionSurface:
    """
    Class constructor
    """
    def __init__(self,
                 time_t: datetime.datetime = None,
                 window: int = 60,
                 asset: str = 'ETH',
                 est_max_maturity: float = 30,
                 est_min_maturity: float = 0.5):
        """
        Initializes the HistoricalOptionSurface which loads required datasets to construct an IV surface at a point
        in time.
        :param time_t: (datetime.datetime) Date at which to fetch historical data from
        :param asset: (str) Name of the underlying asset you want to get option trading data for
                                     ex.: ['ETH', 'BTC', 'SOL']
        :param time_t: (datetime.datetime) Date at which to fetch historical data from
        :param est_min_maturity: (flat) Estimated minimum maturity (in days) of option trades to fetch
        :param est_max_maturity: (float) Estimated maximum maturity (in days) of option trades to fetch

        """
        self.time_t = time_t
        self.allowed_assets = ['BTC', 'ETH', 'SOL', 'USDC']
        self.asset = self.wanted_asset_check(asset)
        self.window = window
        self.est_max_maturity = est_max_maturity
        self.est_min_maturity = est_min_maturity

        self.raw_data = self.get_cleaned_raw_data()

    def wanted_asset_check(self, asset: str) -> str:
        """
        Checks if the asset name is in the allowed assets list.
        :param asset: (list) Name of the underlying asset you want to get option trading data for
        """
        if asset not in self.allowed_assets:
            raise ValueError(f'Invalid asset data requested.'
                             f' Make sure the requested asset is among {self.allowed_assets}')

        return asset

    def get_deribit_raw(self, asset):
        """
        Makes deribit historical api call to fetch option trade data required to construct the historical surface
        """
        window = self.window
        end = (self.time_t + datetime.timedelta(minutes=window/2)).timestamp()*1e3
        start = (self.time_t - datetime.timedelta(minutes=window/2)).timestamp()*1e3

        url = ('https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time?'
               + 'currency=' + asset
               + '&start_timestamp=' + str(int(start))
               + '&end_timestamp=' + str(int(end))
               + '&kind=option'
               + '&count=10000'
               + '&include_old=true')
        resp = requests.get(url).json()

        if 'result' not in list(resp.keys()):
            raise ValueError(f'API Call unsuccessful, error_message: {resp["error"]}')
        elif not resp['result']['trades']:
            raise ValueError('No trades found, trade a wider window or different time_t')

        return resp['result']['trades']

    @staticmethod
    def option_data_processing(raw_data) -> pd.DataFrame:
        """
        Creates an organized pandas DataFrame from the raw options data collected from the
        Deribit API calls.
        :param raw_data: (pd.DataFrame) Dataframe of Raw option trade data.
        :return: (pd.DataFrame) Processed dataframe of option trade data
        """
        dataframe = pd.DataFrame.from_dict(raw_data)
        dataframe.sort_values(by=['timestamp'], inplace=True)

        dataframe['datetime'] = pd.to_datetime(dataframe.timestamp, unit='ms', utc=True)

        instrument_columns = ['asset', 'expiry_date', 'strike', 'option_type']
        dataframe[instrument_columns] = dataframe.instrument_name.str.split('-', expand=True)

        convert_expiry = lambda date: datetime.datetime.strptime(date, '%d%b%y')
        expiry_timestamp = lambda date: datetime.datetime(date.year, date.month, date.day, 8,
                                                          tzinfo=datetime.timezone.utc).timestamp()*1e3

        dataframe.expiry_date = dataframe.expiry_date.apply(convert_expiry)
        dataframe['expiry_timestamp'] = dataframe.expiry_date.apply(expiry_timestamp)

        dataframe.strike = pd.to_numeric(dataframe.strike)

        # Add the datetime index
        dataframe.set_index('datetime', inplace=True)

        return dataframe

    def get_cleaned_raw_data(self) -> pd.DataFrame:
        """
        Gets the processed trade data from the api call and cleans and filters.
        :return:
        """

        api_data = self.get_deribit_raw(self.asset)
        raw_data = self.option_data_processing(api_data)

        # remove extra cols
        drop_columns = ['trade_seq', 'trade_id', 'tick_direction']
        temp_data = raw_data.copy()
        temp_data.drop(columns=drop_columns, inplace=True)

        # remove unneeded expiries
        min_expiries_mask = (temp_data.expiry_timestamp - temp_data.timestamp)/1e3 >= self.est_min_maturity*24*3600
        max_expiries_mask = (temp_data.expiry_timestamp - temp_data.timestamp)/1e3 <= self.est_max_maturity*24*3600

        if (min_expiries_mask & max_expiries_mask).sum() == 0:
            warnings.warn(f'No trades found within desired expiry range, reverting to full set')
            raw_data_cleaned = temp_data.copy()
        else:
            raw_data_cleaned = temp_data.loc[min_expiries_mask & max_expiries_mask].copy()

        return raw_data_cleaned

    def get_option_data(self,
                        option_type: str = None,
                        maturity: int = None,
                        strike: int = None):
        """
        Gets the specific option data from the raw dataframe based on the
        provided specifications
        :param option_type: (str) Option type: put('P') pr call('C')
        :param maturity: (int) Time to maturity.
        :param strike: (int) The strike price of the desired option.
        :return: (pd.DataFrame)
        """

        initial_data = self.raw_data.copy()
        initial_data['ttm'] = round((initial_data.expiry_timestamp - initial_data.timestamp)/1000/3600/24, 0)

        initial_data['iv'] = initial_data['iv'] / 100

        requirements = [option_type, strike, maturity]
        columns = ['option_type', 'strike', 'ttm']

        for req, col in zip(requirements, columns):
            if req is not None:
                initial_data = initial_data.loc[initial_data[col] == req]

        output = initial_data

        #TODO : Add error handling here
        output['mark_iv'] = output.apply(self.calculate_iv, axis=1)

        return output

    @staticmethod
    def calculate_iv(row: pd.Series):
        """
        Determines the mark_iv for a particular row of trade data.
        :param row: (pd.Series) Single row of trade data from raw_data DF.
        :return: (float)
        """

        is_call = (row.option_type == 'C')
        tau = (row.expiry_timestamp-row.timestamp)/1e3/3600/24/365

        mark_iv = get_mark_iv(spot=row.index_price, strike=row.strike, tau=tau, sigma=row.iv,
                              option_price=row.price, mark_price=row.mark_price, is_call=is_call)[0]

        return mark_iv

    def get_iv_data(self, maturity: int = None):
        """
        Gets the Implied Volatility data from a specific time based on the
        provided specifications
        :param maturity: (int) Time to maturity.
        :return: (pd.DataFrame)
        """
        iv_data_raw = self.get_option_data()

        if maturity is not None:

            iv_data = iv_data_raw.groupby('strike',
                                          as_index=False)[['mark_iv', 'iv']].mean()

        else:

            iv_data = iv_data_raw.groupby(['expiry_date', 'ttm', 'strike'],
                                          as_index=False)[['strike', 'mark_iv', 'iv']].mean()

        return iv_data

    def get_iv_surface(self):
        """
        Transforms option trade data into Implied Volatility Surface.
        :return: (pd.DataFrame)
        """
        iv_data_raw = self.get_option_data()

        iv_surface = iv_data_raw.groupby(['expiry_date', 'expiry_timestamp', 'strike'],
                                         as_index=False)[['strike', 'mark_iv']].mean()

        iv_surface['tau'] = (iv_surface.expiry_timestamp/1e3-self.time_t.timestamp())/3600/24/365

        return iv_surface

    @staticmethod
    def strike_interpolation(skew, strike):
        """
        Determines the implied volatility of an option with a given strike price from the IV skew for that maturity
        :param skew: (pd.DataFrame) IV surface object with data from 1 expiry date.
        :param strike: (float) Strike price of option
        :return: (float) Implied volatility of option
        """
        for col in ['strike', 'mark_iv']:
            if col not in skew.columns:
                raise ValueError(f'Skew does not have required column: {col}')

        x = skew.strike.values
        y = skew.mark_iv.values

        if skew.shape[0] == 1:
            iv = y
        else:
            f = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))
            iv = f(strike)

        return iv

    def time_interpolation(self, near_skew, far_skew, tau, strike):
        """
        Determines the IV of an option with given tau and strike price from the two skews that straddle the expiry date
        :param near_skew: (pd.DataFrame) IV surface object with data from 1 expiry date.
        :param far_skew: (pd.DataFrame) IV surface object with data from 1 expiry date.
        :param tau: (float) Time to maturity (in years) of option
        :param strike: (float) Strike price of option
        :return: (float) Implied volatility of option
        """
        for col in ['tau', 'strike', 'mark_iv']:
            if col not in near_skew.columns:
                raise ValueError(f'near_skew does not have required column: {col}')
            if col not in far_skew.columns:
                raise ValueError(f'far_skew does not have required column: {col}')

        near_iv = self.strike_interpolation(near_skew, strike)
        far_iv = self.strike_interpolation(far_skew, strike)

        near_tau = near_skew.tau.iloc[0]
        far_tau = far_skew.tau.iloc[0]

        cum_variance_near = near_iv**2 * near_tau
        cum_variance_far = far_iv**2 * far_tau

        f = interp1d([near_tau, far_tau], [cum_variance_near, cum_variance_far])

        cum_variance = f(tau)
        iv = np.sqrt(cum_variance/tau)

        return iv

    def surface_interpolation(self, iv_surface, tau, strike):
        """
        :param iv_surface: (pd.DataFrame) IV surface object from .get_iv_surface method
        :param tau: (float) Time to maturity (in years) of option
        :param strike: (float) Strike price of option
        :return: (float) IV of option
        """
        for col in ['tau', 'strike', 'mark_iv']:
            if col not in iv_surface.columns:
                raise ValueError(f'IV Surface does not have required column: {col}')

        iv_surface.sort_values(['tau', 'strike'], inplace=True)
        unique_expiries = iv_surface.tau.unique()

        if tau in unique_expiries:
            skew = iv_surface[iv_surface.tau == tau][['strike', 'mark_iv']]
            iv = self.strike_interpolation(skew, strike) + 0

        elif tau < unique_expiries.min():
            skew = iv_surface[iv_surface.tau == unique_expiries.min()][['strike', 'mark_iv']]
            iv = self.strike_interpolation(skew, strike) + 0

        elif tau > unique_expiries.max():
            skew = iv_surface[iv_surface.tau == unique_expiries.max()][['strike', 'mark_iv']]
            iv = self.strike_interpolation(skew, strike) + 0

        else:
            less_than_mask = unique_expiries < tau
            near_expiry_idx = np.where(less_than_mask)[0][-1]

            greater_than_mask = unique_expiries > tau
            far_expiry_idx = np.where(greater_than_mask)[0][0]

            near_skew = iv_surface[iv_surface.tau == unique_expiries[near_expiry_idx]][['strike', 'mark_iv']]
            far_skew = iv_surface[iv_surface.tau == unique_expiries[far_expiry_idx]][['strike', 'mark_iv']]

            iv = self.time_interpolation(near_skew, far_skew, tau, strike)

        return iv
