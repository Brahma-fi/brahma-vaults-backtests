"""
A module for aggregating the data from the off-chain sources. For the
perp futures data there is an option of adding funding rates data.
#####################################################################
Data format
SPOT MARKET:    'ETHUSDT' or 'ethusdt'
PERP MARKET:    'ALTUSDT-PERP' or 'altusdt-perp'
FUTURES MARKET: 'BTCUSD-CURRENT' or 'btcusd-next'
DVOL DATA:      'ETH-DVOL' or 'eth-dvol'

OPTION DATA: 'ETH-OPT' or 'eth-opt'
"""

# pylint: disable=anomalous-backslash-in-string, too-many-instance-attributes, too-many-public-methods, too-many-arguments, too-many-locals, invalid-name
import re
import datetime
import itertools
from typing import Union, List, Tuple, Dict, Any
import requests
import numpy as np
import pandas as pd


class DeFibulizer():
    """
    Class for data aggregation and processing of the on-chain and off-chain data.
    """

    def __init__(self):
        """
        Class constructor.
        """

        self.asset_names = None
        self.asset_types = None
        self.start_date = None
        self.end_date = None
        self.frequency = None
        self.get_funding_rate = None
        self.joint_dataframe = None
        self.allowed_dvol = ['ETH', 'BTC']

    @staticmethod
    def set_asset_name(asset_names: Union[str, List[str]]) -> np.ndarray:
        """
        Checks whither the provided asset_names parameter type is correct and
        transforms the data into an uppercase list of strings if needed.
        :param asset_names: (str, List[str]) Provided list of asset names to
                            retrieve data for.
        :return: (List[str]) List of the asset names to retrieve data for.
        """
        # Performing the typing check and transformation.

        if isinstance(asset_names, str):

            # If the singular asset is provided transforming it into the list.
            asset_names = np.array([asset_names.upper()])

        elif isinstance(asset_names, list) and all(isinstance(asset, str) for asset in asset_names):

            # If multiple assets are provided in a form of the list keeping it intact.
            asset_names = np.array([name.upper() for name in asset_names])

        else:

            # Else raise type error.
            raise TypeError('Incorrect asset type, please provide either str or List[str].')

        return asset_names

    @staticmethod
    def set_date(date: Union[datetime.datetime, str]) -> int:
        """
        Checks whither the provided dates parameter types are correct and
        transforms the data into an timestamps.
        :param date: (datetime.datetime/str)
        :return: (datetime.timestamp) Timestamp to retrieve data for.
        """
        # Performing the type check and transformation.

        if isinstance(date, str):
            # Splitting the string into year month and day component.
            temp = date.split('-')

            # Transform the date into timestamp.
            date = datetime.datetime(int(temp[0]), int(temp[1]), int(temp[2])).timestamp()

        elif isinstance(date, datetime.datetime):

            # If multiple assets are provided in a form of the list keeping it intact.
            date = date.timestamp()
        else:

            # Else raise type error.
            raise TypeError('Incorrect date type, please provide either YYYY-MM-DD '
                            'or datetime.datetime.')

        return int(date)

    @staticmethod
    def set_frequency(frequency: str = 'h') -> Tuple[int, str]:
        """
        Checks whether the provided frequency belongs to the range of available options
        and convert it into seconds.
        Options:
        m - minutely
        h - hourly
        d - daily
        :param frequency: (str) Desired data frequency in str format.
        :return: (int) Data frequency in seconds.
        """
        # Initializing
        freq_str = ['m', 'h', 'd']
        freq_seconds = [60, 3600, 86400]

        adjust_for_binance = lambda el: '1' + el

        if frequency not in freq_str:
            # Raise type error if the frequency provided does not belong to the
            # list of possible options.
            raise TypeError('Incorrect frequency type, please provide either m, h or d.')

        # Finding the corresponding index in the freq_str
        index = freq_str.index(frequency)

        # Taking the equivalent value in seconds from the frec_seconds
        frequency = (freq_seconds[index], adjust_for_binance(freq_str[index]))

        return frequency

    def is_spot(self) -> list:
        """
        Creates a boolean mask that returns true values if the asset belongs to spot market
        based on the provided asset symbol formatting.
        :return: (list) List of boolean indexes that showcase whether the asset
                        belongs to the spot market.
        """
        # Creating a regular expression pattern to recognize the assets from spot market.
        spot_pattern = re.compile('^([A-Z]+)$')

        # Applying the pattern to the asset names to create the boolean mask
        spot_mask = [bool(spot_pattern.match(name)) for name in self.asset_names]

        return spot_mask

    def is_perp(self) -> list:
        """
        Creates a boolean mask that returNs true values if the asset belongs to future market
        and is a perpetual based on the provided asset symbol formatting.
        :return: (list) List of boolean indexes that showcase whether the asset belongs to
                        the spot market.
        """
        # Creating a regular expression pattern to recognize the assets from spot market.
        perp_pattern = re.compile('^([A-Z]+)[\-](PERP)$')

        # Applying the pattern to the asset names to create the boolean mask
        perp_mask = [bool(perp_pattern.match(name)) for name in self.asset_names]

        return perp_mask

    def is_futures(self) -> list:
        """
        Creates a boolean mask that returns true values if the asset belongs to future market
        and has an expiry date based on the provided asset symbol formatting.
        :return: (list) List of boolean indexes that showcase whether the asset belongs
                        to the spot market.
        """
        # Creating a regular expression pattern to recognize the assets from spot market.
        futures_pattern = re.compile('^([A-Z]+)[\-](CURRENT|NEXT)$')

        # Applying the pattern to the asset names to create the boolean mask
        futures_mask = [bool(futures_pattern.match(name)) for name in self.asset_names]

        return futures_mask

    def is_dvol(self) -> list:
        """
        Creates a boolean mask that returns true values if the asset asset_name reffers to DVOL
        data based on the provided asset symbol formatting.
        :return: (list) List of boolean indexes that showcase whether the asset belongs
                        to the spot market.
        """
        # Creating a regular expression pattern to recognize the assets from spot market.
        dvol_pattern = re.compile('^([A-Z]+)[\-](DVOL)$')

        # Applying the pattern to the asset names to create the boolean mask
        dvol_mask = [bool(dvol_pattern.match(name)) for name in self.asset_names]

        return dvol_mask

    def is_option(self) -> list:
        """
        Creates a boolean mask that returns true values if the asset asset_name reffers to option
        data based on the provided asset symbol formatting.

        :return: (list) List of boolean indexes that showcase whether the asset belongs
                        to the spot market.
        """
        # Creating a regular expression pattern to recognize the assets from spot market.
        dvol_pattern = re.compile('^([A-Z]+)[\-](OPT)$')

        # Applying the pattern to the asset names to create the boolean mask
        dvol_mask = [bool(dvol_pattern.match(name)) for name in self.asset_names]

        return dvol_mask

    def asset_mapping(self):
        """
        Maps the provided asset/assets to the class of asset it belongs to, returning the
        array of mapped values.
        Mappings:
        s - spot market assets
        p - perpetual futures
        f - regular futures
        v - volatility index
        o - options

        :return: (list) Asset mapping to the asset types.
        """
        # Initializing the array for mapped values.
        asset_types = self.asset_names.copy()

        # Assigning the list of possible labels for the assets.
        asset_labels = ['s', 'p', 'f', 'v', 'o']

        # Creating the boolean masks for each type of assets.
        asset_masks = [self.is_spot(), self.is_perp(), self.is_futures(), self.is_dvol(), self.is_option()]

        # Mapping the labels.
        for label, mask in zip(asset_labels, asset_masks):
            asset_types[mask] = label

        for index, element in enumerate(asset_types):

            if element not in asset_labels:
                raise ValueError(f'The {index} element asset_name is provided incorrectly')

        return asset_types

    def get_binance_spot_query(self, asset_name: str) -> Dict[str, Union[Union[str, int], Any]]:
        """
        Prepares the provided general data and the asset_name to transform into
        the required format for the Binance spot data request.
        :param asset_name: (str) The asset_name of the asset.
        :return: (list) Query details for the Binance data.
        """

        start_date = int(self.start_date * 1e3)
        end_date = int(self.end_date * 1e3)
        interval = self.frequency[1]
        limit = 1000

        query = {'symbol': asset_name,
                 'interval': interval,
                 'startTime': start_date,
                 'endTime': end_date,
                 'limit': limit
                 }

        return query

    def get_binance_futures_query(self, asset_name: str, asset_type: str):
        """
        Prepares the provided general data and the asset_name to transform into
        the required format for the Binance futures data request.
        :param asset_name: (str) The asset_name of the asset.
        :param asset_type: (str) The type of the futures - finite or perpetual.
        :return: (list) Query details for the Binance data.
        """
        # Setup the initial data parameters
        start_date = int(self.start_date * 1e3)
        end_date = int(self.end_date * 1e3)
        limit = 1000
        interval = self.frequency[1]
        name, contract_type = asset_name.split('-')

        # Complete the string for the contract type
        if asset_type == 'p':
            string_completion = 'ETUAL'
        elif asset_type == 'f':
            string_completion = '_QUARTER'
        else:
            raise ValueError('Incorrect asset type literal.')

        contract_type += string_completion

        query = {'pair': name,
                 'contractType': contract_type,
                 'interval': interval,
                 'startTime': start_date,
                 'endTime': end_date,
                 'limit': limit
                 }

        return query

    def get_binance_funding_query(self, asset_name: str):
        """
        Prepares the provided general data and the asset_name to transform into
        the required format for the Binance funding rate data request.
        :param asset_name: (str) The asset_name of the asset.
        :return: (list) Query details for the Binance data.
        """
        # Setup the initial data parameters
        start_date = int(self.start_date * 1e3)
        end_date = int(self.end_date * 1e3)
        limit = 1000
        name, _ = asset_name.split('-')

        query = {'symbol': name,
                 'startTime': start_date,
                 'endTime': end_date,
                 'limit': limit
                 }

        return query

    @staticmethod
    def get_url(query: dict, data_type: str) -> List[Union[str, dict]]:
        """
        Creates url string to be used in API call
        :param query: (dict) Dict of the key query parameters.
        :param data_type: (str) The type of the URL request to construct.
                               Input options: 'binance_spot'; 'binance_funding'; 'deribit_dvol'
        :return: (list) URL for the data request and a dict of request parameters.
        """
        params = query

        if data_type == 'binance_spot':
            # Constructing an URL for the spot/futures/perp price data request
            url = 'https://api.binance.com/api/v3/klines'

        if data_type == 'binance_futures':
            url = 'https://fapi.binance.com/fapi/v1/continuousKlines'

        elif data_type == 'binance_funding':

            # Constructing an URL for the funding rate request
            url = 'https://fapi.binance.com/fapi/v1/fundingRate'

        elif data_type == 'deribit_dvol':

            # Constructing an URL for the DVOL price data request
            url = 'https://www.deribit.com/api/v2/public/get_volatility_index_data?'

        elif data_type == 'deribit_option':

            # Constructing an URL for the Derobit option trade data request
            url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time?'

        else:

            # Else raise type error.
            raise TypeError('Incorrect URL type')

        output = [url, params]

        return output

    def get_deribit_query(self, name: str, asset_type: str) -> Dict[str, Union[Union[str, int, float], Any]]:
        """
        Prepares the provided general data and the asset_name to transform into
        the required format for the Deribit data.
        :param name: (str) The asset_name of the asset.
        :param asset_type: (str) The type of the deribit data to request - dvol or options.
        :return: (list) Query details for the Deribit data.
        """
        # Select only the part relevant to the query
        name = name.split('-')[0]

        # Adjust the timestamp to be in ms
        start_date = int(self.start_date * 1e3)
        end_date = int(self.end_date * 1e3)

        frequency = self.frequency[0]

        # Adjust the symbol for daily data
        if frequency == 86400:
            frequency = '1D'

        # Set up the base deribit query parameters
        query = {'currency': name,
                 'start_timestamp': start_date,
                 'end_timestamp': end_date
                 }

        # Add asset-specific query parameters
        if asset_type == 'v':
            query['resolution'] = frequency

        elif asset_type == 'o':
            query['kind'] = 'option'
            query['count'] = 1e4
            query['include_old'] = 'true'

        else:
            raise ValueError('Incorrect asset type literal.')

        # query = [name, start_date, end_date, frequency]

        return query

    def get_fractional_binance_request(self, query: dict, data_type: str) -> list:
        """
        Generator function that generates Binance request results for the fractions of
        the data range until the whole data range is processed.
        :param query: (list) The list of API query parameters:
                             [asset_names, start_date, end_date, frequency].
        :param data_type: (str) The type of data that needs to be downloaded 'binance_spot'or
                                'binance_funding'
        """
        # Getting the start and end date from the initial query
        start_date = query['startTime']
        end_date = query['endTime']

        # Establishing an appropriate step for the request, -1 to avoid jumping over the 9th hours
        step = int(query['limit'] * self.frequency[0] * 1e3 - 1)

        # Starting a cycle of generating the outcomes of fractional requests until
        # the whole data range is processed


        for start_time in range(start_date, end_date, step):

            # Calculating an end_time of a fraction
            end_time = min(start_time + step, end_date)

            # Set the new start/end time for the fractional query
            query['startTime'] = start_time
            query['endTime'] = end_time
            # Creating an URL for the request
            url, params = self.get_url(query, data_type)

            # Trying getting the results of the query
            resp = requests.get(url, params=params).json()

            if isinstance(resp, dict):
                msg = 'msg'
                raise ValueError(f'{resp[msg]}')

            # Yielding the request results
            yield resp

    def get_fractional_deribit_request(self, query: dict) -> list:
        """
        Generator function that generates deribit request results for the fractions
        of the data range until the whole data range is processed.
        :param query: (list) The list of API query parameters:
                             [asset_names, start_date, end_date, frequency].
        """
        # Setting the continuation as the end date
        continuation = query['end_timestamp']
        currency = 'currency'

        while continuation is not None:

            query['end_timestamp'] = continuation

            # Creating an URL for the request
            continuation_url, params = self.get_url(query, 'deribit_dvol')

            # Trying to get the request and return an exception otherwise
            try:
                resp = requests.get(continuation_url, params=params).json()['result']
            except TypeError as type_err:  # pragma: no cover
                raise TypeError(f'The provided {query[currency]} deribit data is incorrect.') from type_err
            except KeyError as key_err:  # pragma: no cover
                raise KeyError(requests.get(continuation_url, params=params).json()['error']) from key_err

            data = resp['data']

            yield data

            continuation = resp['continuation']  # pragma: no cover

    def get_fractional_deribit_option_request(self, query: dict) -> list:
        """
        Generator function that generates deribit request option trades results for the fractions
        of the data range until the whole data range is processed.

        :param query: (list) The list of API query parameters:
                             [asset_names, start_date, end_date, frequency].
        """
        # Setting the continuation as the end date
        continuation = query['start_timestamp']

        currency = 'currency'

        has_more = True

        while has_more:

            query['start_timestamp'] = continuation

            # Creating an URL for the request
            continuation_url, params = self.get_url(query, 'deribit_option')

            # Trying to get the request and return an exception otherwise
            try:
                resp = requests.get(continuation_url, params=params).json()['result']
            except TypeError as type_err:  # pragma: no cover
                raise TypeError(f'The provided {query[currency]} option data is incorrect.') from type_err
            except KeyError as key_err:  # pragma: no cover
                raise KeyError(requests.get(continuation_url, params=params).json()['error']) from key_err

            data = resp['trades']

            yield data

            continuation = resp['trades'][-1]['timestamp']  # pragma: no cover

            has_more = resp['has_more']

    def get_binance_data(self, query: dict, data_type: str) -> list:
        """
        Retrieves the raw json data from the Binance API according to passed query
        parameters. The query provided has to include the asset asset_name, start and end
        dates for the required data and the frequency. The retrieved data is then
        transformed into a single list.
        :param query: (list) The list of API query parameters:
                             [asset_names, start_date, end_date, frequency].
        :return: (list) List of data for the provided query date range, data format:
                        [startTime, time, open, high, low, close, volume].
        """

        # Get the Binance request generator
        binance_generator = self.get_fractional_binance_request(query=query,
                                                                data_type=data_type)

        # Create a list from the generator for the Binance requests
        output = list(itertools.chain.from_iterable(binance_generator))

        return output

    def get_deribit_data(self, query: dict, is_option: bool = True) -> dict:
        """
        Retrieves the raw vol data from the Deribit API according to passed query
        parameters. The query provided has to include the asset asset_name, start and end dates for
        the required data and the frequency. The retrieved data is then transformed into
        a single list.
        :param query: (list) The list of API query parameters:
                             [asset_names, start_date, end_date, frequency]
        :return: (list) List of data for the provided query date range, data format:
                        [time, open, high, low, close].
        """
        # Check whether the token has data available
        if not query['currency'] in self.allowed_dvol:
            raise ValueError('The dvol data for this asset is not available')

        # Getting the deribit request generator
        if is_option:
            deribit_generator = self.get_fractional_deribit_option_request(query=query)
        else:
            deribit_generator = self.get_fractional_deribit_request(query=query)

        # Getting the full list of data for the whole range of dates from generator
        unchained_output = list(itertools.chain.from_iterable(deribit_generator))
        output = list(itertools.chain(unchained_output))

        return output

    @staticmethod
    def dvol_data_processing(raw_data: dict, asset_names: str) -> pd.DataFrame:
        """
        Creates an organized pandas DataFrame from the raw data collected from the
        Deribit API calls.
        :param raw_data: (pd.DataFrame) Raw list of dictionaries containing the API request output.
        :param asset_names: (str) The asset_name of the asset.
        :return: (pd.DataFrame) Query details for the Deribit data.
        """

        # Creating a pd.DataFrame from the raw data
        dataframe = pd.DataFrame(raw_data)

        # Establishing the column names
        dataframe.columns = ['time', 'open', 'high', 'low', 'close']

        dataframe = dataframe.sort_values(by=['time'])

        # Add the datetime index
        timestamp_temp = pd.to_datetime(dataframe.time, unit='ms', utc=True)

        dataframe.set_index(timestamp_temp, inplace=True)

        # Add a MultiIndex
        dataframe.columns = pd.MultiIndex.from_product([[asset_names], dataframe.columns])

        return dataframe

    @staticmethod
    def option_data_processing(raw_data: dict) -> pd.DataFrame:
        """
        Creates an organized pandas DataFrame from the raw options data collected from the
        Deribit API calls.

        :param raw_data: (pd.DataFrame) Raw list of dictionaries containing the API request output.

        :return: (pd.DataFrame) Query details for the Deribit data.
        """

        # Creating a pd.DataFrame from the raw data
        dataframe = pd.DataFrame.from_dict(raw_data)

        # Establishing the column names
        temp_timestamp = pd.to_datetime(dataframe.timestamp, unit='ms')

        dataframe = dataframe.sort_values(by=['timestamp'])

        temp = dataframe.instrument_name.str.split('-').tolist()

        dataframe[['asset', 'expiry_date', 'strike', 'option_type']] = pd.DataFrame(temp, index=dataframe.index)

        convert_expiry = lambda date: datetime.datetime.strptime(date, '%d%b%y')

        dataframe.expiry_date = dataframe.expiry_date.apply(convert_expiry)

        dataframe.strike = pd.to_numeric(dataframe.strike)

        # Add the datetime index

        dataframe.set_index(temp_timestamp, inplace=True)

        return dataframe

    @staticmethod
    def binance_data_processing(raw_data: list,
                                asset_name: str,
                                raw_funding_rates=None) -> pd.DataFrame:
        """
        Creates an organized pandas DataFrame from the raw data collected from the
        Binance API calls.
        :param raw_data: (pd.DataFrame) Raw list of dictionaries containing the API request output
        :param asset_name: (str) The asset_name of the asset.
        :param raw_funding_rates: (pd.DataFrame) Raw list of dictionaries containing the API
                                                 request output for the funding rates.
        :return: (pd.DataFrame) Query details for the Deribit data.
        """

        # Creating a pd.DataFrame from the raw data
        dataframe = pd.DataFrame(raw_data)

        columns = ['open_time', 'open', 'high',
                   'low', 'close', 'volume',
                   'close_time', 'quote_asset_volume',
                   'number_of_trades', 'taker_buy_volume',
                   'taker_buy_quote_asset_volume', 'drop']

        dataframe.columns = columns

        convert_to_date = lambda col: pd.to_datetime(col, unit='ms', utc=True)

        if raw_funding_rates is not None:
            # Creating the funding rates dataframe
            funding_dataframe = pd.DataFrame(raw_funding_rates)
            # Merging the funding data with the original perp dataframe
            dataframe = dataframe.merge(funding_dataframe, how='left',
                                        left_on='open_time', right_on='fundingTime')

            dataframe.drop(['fundingTime', 'symbol'], axis=1, inplace=True)

        dataframe.drop(['drop'], axis=1, inplace=True)

        time_related_subset = dataframe.filter(regex='time')
        not_time_subset = dataframe.filter(regex='^((?!time).)*$')

        dataframe[time_related_subset.columns] = time_related_subset.apply(convert_to_date)

        # Add the datetime index
        timestamp_temp = pd.to_datetime(dataframe.open_time, unit='ms', utc=True)

        dataframe.set_index(timestamp_temp, inplace=True)

        dataframe[not_time_subset.columns] = dataframe[not_time_subset.columns].astype(float)

        # Adding a MultiIndex
        dataframe.columns = pd.MultiIndex.from_product([[asset_name], dataframe.columns])

        return dataframe

    def get_spot_data(self, data: list, asset_name: str):
        """
        Runs the procedure to form a request, get and postprocess the spot
        data. Then the dataframe is appended to the list of the dataframes.

        :param data: (list) The list of dataframes for all the retrieved data.
        :param asset_name: (str) The name of the asset to retrieve.
        """
        # Creating a query variables list
        query = self.get_binance_spot_query(asset_name)
        # Getting the raw data from the Binance API
        raw_data = self.get_binance_data(query, 'binance_spot')
        # Appending the processed dataframe
        data.append(self.binance_data_processing(raw_data, asset_name))

    def get_perp_data(self, data: list, asset_name: str):
        """
        Runs the procedure to form a request, get and postprocess the perpetual futures
        data. Then the dataframe is appended to the list of the dataframes.

        :param data: (list) The list of dataframes for all the retrieved data.
        :param asset_name: (str) The name of the asset to retrieve.
        """
        asset_type = 'p'

        # Creating a query variables list for the perp
        query = self.get_binance_futures_query(asset_name, asset_type)
        # Getting the raw data from the Binance API
        raw_data = self.get_binance_data(query, 'binance_futures')

        # If there is a flag for getting the funding rate start the procedure
        if self.get_funding_rate:
            # Creating a query variables list for the perp funding rates
            query = self.get_binance_funding_query(asset_name)
            # Getting the raw funding rates data from the Binance API
            raw_funding_rates = self.get_binance_data(query, 'binance_funding')
            # Appending the processed dataframe
            data.append(self.binance_data_processing(raw_data, asset_name, raw_funding_rates))

        else:
            # Appending the processed dataframe
            data.append(self.binance_data_processing(raw_data, asset_name))

    def get_futures_data(self, data: list, asset_name: str):
        """
        Runs the procedure to form a request, get and postprocess the finite futures
        data. Then the dataframe is appended to the list of the dataframes.

        :param data: (list) The list of dataframes for all the retrieved data.
        :param asset_name: (str) The name of the asset to retrieve.
        """
        asset_type = 'f'

        # Creating a query variables list for the future
        query = self.get_binance_futures_query(asset_name, asset_type)
        # Getting the raw data from the Binance API
        raw_data = self.get_binance_data(query, 'binance_futures')
        # Appending the processed dataframe
        data.append(self.binance_data_processing(raw_data, asset_name))

    def get_options_data(self, data: list, asset_name: str):
        """
        Runs the procedure to form a request, get and postprocess the Deribit options
        data. Then the dataframe is appended to the list of the dataframes.

        :param data: (list) The list of dataframes for all the retrieved data.
        :param asset_name: (str) The name of the asset to retrieve.
        """
        asset_type = 'o'
        # Creating a query variables list for the future
        query = self.get_deribit_query(asset_name, asset_type)
        # Getting the raw data from the Deribit API
        raw_data = self.get_deribit_data(query, is_option=True)
        # Appending the processed dataframe
        data.append(self.option_data_processing(raw_data))

    def get_dvol_data(self, data: list, asset_name: str):
        """
        Runs the procedure to form a request, get and postprocess the volatility index
        data. Then the dataframe is appended to the list of the dataframes.

        :param data: (list) The list of dataframes for all the retrieved data.
        :param asset_name: (str) The name of the asset to retrieve.
        """
        asset_type = 'v'

        # Creating a query variables list for the future
        query = self.get_deribit_query(asset_name, asset_type)
        # Getting the raw data from the Deribit API
        raw_data = self.get_deribit_data(query, is_option=False)
        # Appending the processed dataframe
        data.append(self.dvol_data_processing(raw_data, asset_name))

    def init_asset_getter_dict(self):
        """
        Creates a dictionary of data retrieving procedures corresponding to
        respective types of assets.

        :return: (dict) Dictionary with the asset types as key and processing
                        functions as values.
        """
        asset_types = ['s', 'p', 'f', 'v', 'o']

        data_getters = [self.get_spot_data,
                        self.get_perp_data,
                        self.get_futures_data,
                        self.get_dvol_data,
                        self.get_options_data]

        output = dict(map(lambda i, j: (i, j), asset_types, data_getters))

        return output

    def get_data(self,
                 asset_names: Union[str, List[str]] = None,
                 start_date: Union[datetime.datetime, str] = None,
                 end_date: Union[datetime.datetime, str] = None,
                 frequency: str = 'm',
                 get_funding_rate: bool = False,
                 joint_dataframe=True,
                 save_csv: bool = False) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Gathers the available instrument data in a requested data range for every
        instance in the provided asset_names list. The data is gathered using the
        respective API - Binance or Deribit. Obtained raw data is then processed into
        a single concatenated dataframe for all assets/list of data frames for
        each individual asset data.
        :param asset_names: (str, List[str]) List of asset names to retrieve data for.
        :param start_date: (datetime.datetime, str) Beginning of data range. Str format
                                                    should be YYYY-MM-DD.
        :param end_date: (datetime.datetime, str) End of the data range. tr format
                                                    should be YYYY-MM-DD.
        :param frequency: (str) The wanted data frequency. Available formats:
                                'm' - minutely; 'h' - hourly; 'd' - daily.
        :param get_funding_rate: (bool) Flag whether to retrieve the funding rate data
                                        for the perpetual futures.
        :param joint_dataframe: (bool) Flag whether to return a joint dataframe or a list
                                       of separate dataframes.
        :param save_csv: (bool) Flag whether to save the dataframe on users system.
        :return: (pd.DataFrame, List[pd.DataFrame]) Processed data.
        """
        # Setting all the attributes from the provided arguments
        self.asset_names = self.set_asset_name(asset_names)
        # Mapping the asset names to the asset type
        self.asset_types = self.asset_mapping()
        # Transforming the start and end date into int(timestamps)
        self.start_date = self.set_date(start_date)
        self.end_date = self.set_date(end_date)
        # Setting the data frequency
        self.frequency = self.set_frequency(frequency)
        # Setting the flags rom the provided
        self.get_funding_rate = get_funding_rate
        self.joint_dataframe = joint_dataframe

        # Checking whether the start and end dates are valid
        if self.start_date > self.end_date:
            raise ValueError('End date can not precede the start date.')

        # Initializing the data list
        data = []

        fill_in_parameters = lambda func: func(data=data, asset_name=asset_name)

        # Initialize the dictionary of a getter functions for respective asset types
        data_getter = self.init_asset_getter_dict()

        if 'o' in self.asset_types:
            joint_dataframe = False

        # For every asset in provided list get the data from the respective API
        for asset_type, asset_name in zip(self.asset_types, self.asset_names):

            # If the asset belongs to spot market, perpetual futures, finite futures
            # options or Deribit volatility indexes - start the respective procedure
            fill_in_parameters(data_getter[asset_type])

        # Concatenating the dataframes if the flag is true
        if joint_dataframe:
            for el in data:
                ind = el[el.index.duplicated()].index
                el.drop(ind, axis=0, inplace=True)
            output_data = pd.concat(data, axis=1)

            output_data.index.name = 'time'

            # Saving the data if the flag is true
            self.save_csv(data, save_csv)
        else:
            output_data = data

        return output_data

    def save_csv(self, data: Union[list, pd.DataFrame] = None, flag: bool = True, test: bool = False) -> None:
        """
        Saves data in a *.csv data format.
        :param data: (list/pd.DataFrame) Data to be saved.
        :param flag: (bool) A flag signifying whether to save the files
        :param test: (bool) A flag signifying whether it is in a unit test or not.
        """
        if flag:
            if isinstance(data, pd.DataFrame):
                if not test:  # pragma: no cover
                    dataframe_name = '_'.join(self.asset_names)
                    data.to_csv(f'{dataframe_name}.csv')

            else:
                # If data is a list - save as separate entities
                if not test:  # pragma: no cover
                    for name, dataframe in zip(self.asset_names, data):
                        dataframe.to_csv(f'{name}.csv')

    @staticmethod
    def load_csv(file_name: str) -> pd.DataFrame:
        """
        Assists the users with reading the MultiIndex csv DataFrame.
        :param file_name: (str) Name of the saved DeFibulizer dataframe or the path to the file.
        :return: (pd.DataFrame) The correctly formatted dataframe from the saved csv file.
        """
        # Reading the csv specifying the levels and indexes
        output = pd.read_csv(file_name, header=[0, 1], index_col=0)

        return output
