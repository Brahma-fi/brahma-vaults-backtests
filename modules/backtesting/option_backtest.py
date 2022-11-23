"""
Option backtest module.
"""
import os
import json
from datetime import timedelta
import numpy as np
import pandas as pd


from modules.backtesting.base_backtest import BaseBacktest
from modules.options.basic_option_helpers import black_scholes_price, surface_interpolation
from modules.util.backtest_helpers import get_str_date
from modules.defibulizer.option_defibulizer import HistoricalOptionSurface


class OptionBacktest(BaseBacktest):
    """
    Option backtest class.
    """

    def __init__(self, price_data: pd.DataFrame, position_data: pd.DataFrame,
                 rate: float, option_rate: float, bid_offer_spread: float,
                 trade_freq: int = 7, strike_otm: float = 0,
                 volatility_bounds: dict = None,
                 strike_rounding: bool = True, is_usd: bool = True,
                 dvol_data: pd.DataFrame = None, volatility_data: pd.DataFrame = None,
                 historical_vol: bool = False):
        """
        Class constructor. Also initializes the whole backtesting process in one run.
        :param price_data: (pd.DataFrame) Hourly OHLC price data.
        :param position_data: (pd.Series) Hourly position data.
        :param rate: (float) Compounding rate for the vault.
        :param trade_freq: (int) Trading frequency in days: 7, 3, etc.
        :param leverage: (int) The amount of leverage used for trades.
        :param strike_otm: (float) Fraction OTM of strike to buy/sell. O corresponds to ATM prices.
        :param volatility_bounds: (dict) A dictionary containing volatility bounds. Possible keys for the boundries:
                                  {'rv_open': [None, None], 'dvol_open': [None, None],
                                   'sigma_open': [None, None], 'dvol_premium': [None, None]}
        :param strike_rounding: (bool) Whether the strike prices need to be rounded.
        :param is_usd: (bool) Are the trades made in one currency or converted between multiple.
        :param dvol_data: (pd.DataFrame) Hourly OC DVOL data.
        :param volatility_data: (pd.DataFrame) Hourly volatility data.
        :param historical_vol: (bool) Flag signifying whether to use the historical vol gathered from deribit
        """
        # Initialize basic backtest parameters

        self.dvol_data_flag = (dvol_data is not None)
        self.volatility_bounds = volatility_bounds

        super().__init__(price_data=price_data, position_data=position_data,
                         trade_freq=trade_freq)

        start_date = get_str_date(self.raw_data.index[0])
        end_date = get_str_date(self.raw_data.index[-1])

        # Setting up volatility values
        self.historical_vol_flag = historical_vol

        path = f'vol_data/historical_vol_data_{start_date}_{end_date}.json'

        # Initializing the
        if self.historical_vol_flag:
            try:
                self.historical_vol_path = path
            except FileNotFoundError:
                self.historical_vol_path = self.create_historical_vol_file(path=path, asset='ETH')
            self.historical_vol_data = self.load_historical_vol()

        if volatility_data is not None:
            volatility_data = volatility_data + bid_offer_spread / 2
        self.raw_data = self.init_raw_with_vol(dvol_data=dvol_data, volatility_data=volatility_data)

        # Setting up the backtest values
        self.margin = 0.0625
        self.option_rate = option_rate
        self.rate = rate * self.trade_freq / 365
        self.strike_otm = strike_otm
        self.strike_rounding = strike_rounding

        self.is_usd = is_usd
        self.transaction_cost = self.get_transaction_cost()

        # Get backtest results
        self.backtest_results = self.get_backtest_results()
        self.strategy_metrics = self.get_strategy_metrics()
        self.signal_metrics = self.get_signal_metrics()

    @property
    def historical_vol_path(self):
        """
        Getter for the historical volatility path.
        :return: (str) Path to the location of the historical volatility file.
        """
        return self._historical_vol_path

    @historical_vol_path.setter
    def historical_vol_path(self, path: str):
        """
        Setter for a historical volatility path
        :param path: (str) Path to the location of the historical volatility file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError('File doesn\'t exist at given path.Adjust the provided path value'
                                    'or use create_historical_vol_file() to create one')

        self._historical_vol_path = path

    def load_historical_vol(self):
        """
        Loads the historical implied volatility data from the file set in the historical vol path.

        :return: (dict) Dictionary of volatility surfaces corresponding to a certain timestamp.
        """
        with open(self.historical_vol_path) as file:

            raw_vol_data = json.load(file)

            output = {}

            # For every timestamp convers the surface data into a pd.Dataframe
            for timestamp in raw_vol_data:
                output[int(timestamp)] = pd.DataFrame.from_dict(raw_vol_data[timestamp])

        return output

    def init_raw_with_vol(self, dvol_data: pd.DataFrame = None,
                          volatility_data: pd.DataFrame = None):
        """
        Initializes raw data with the addition of explicitly provided volatility-related dataframes.
        :param dvol_data: (pd.Dataframe) Dataframe with hourly DVOL index data.
        :param volatility_data: (pd.Dataframe) Dataframe with hourly iv data
        :return: (pd.DataFrame) Dataframe with the
        """
        data = [self.raw_data.copy(), volatility_data, dvol_data, self.get_realised_vol()]

        for i, dataframe in enumerate(data):
            if dataframe is None:
                data.pop(i)

        output = pd.concat(data, axis=1)

        output['dvol_premium'] = output['dvol_open'] - output['rv_start']

        return output

    def get_realised_vol(self):
        """
        Calculates the realised volatility based on the provided hourly ohlc data.
        RV window is based on trade frequency.
        :return: (pd.DataFrame) Raw OHLC data combined with the calculated RV values.
        """

        output = pd.DataFrame(index=self.raw_data.index)

        open_data = self.raw_data.open.copy().resample('1d', offset='8h', label='right').last()

        rv_window = str(self.trade_freq) + 'd'

        output['rv_end'] = np.log(1 + open_data.pct_change()).rolling(rv_window).std() * np.sqrt(365)
        output['rv_start'] = output['rv_end'].shift(24 * self.trade_freq).fillna(method='bfill')

        return output

    def historical_vol_generator(self, asset: str = None):
        """
        Creates a generator object that retrieves the volatility surfaces for a certain
        date range.
        :param asset: (str) The asset you retrieve the option data for.
        :return: (list) Key value pair of timestamp and dict with a volatility surface data.
        """
        raw_open_dates = self.raw_data.position_open_time.copy()

        resampled_dates = raw_open_dates.resample(self.trade_freq,
                                                  offset='8h',
                                                  label='left').agg({'position_open_time': 'first'})

        if asset is None:
            asset = 'ETH'

        for time_iter in resampled_dates.values:
            odz = HistoricalOptionSurface(time_t=time_iter,
                                          window=30,
                                          asset=asset,
                                          est_max_maturity=self.trade_freq + 1,
                                          est_min_maturity=self.trade_freq - 1)
            volatility_surface = odz.get_iv_surface()

            key = int(time_iter.timestamp())
            values = volatility_surface[['strike', 'mark_iv', 'tau']].copy().to_dict()

            output = [key, values]

            yield output


    def create_historical_vol_file(self, path: str = None, asset: str = None):
        """
        Creates a new historical volatility json file at provided path.
        If there is no path provided then the path property is used by default.
        :param path: (str) Path to the file location.
        :param asset: (str) Underlying for options
        :return: (str) Path to the file location.
        """

        if path is None:
            path = self.historical_vol_path

        # Creating volatility dataset dictionary from generator object
        vol_dict = dict(self.historical_vol_generator(asset))
        # Encoding the dictionary
        vol_dict_encoded = json.dumps(vol_dict)

        with open(path, 'w') as output_file:

            print(vol_dict_encoded, file=output_file)

        self.historical_vol_path = path
        output = path

        return output

    @staticmethod
    def get_implied_vol(row: pd.Series, tau: float, historical_vol_data: dict):
        """
        Iteratively calculates the implied volatility for each row of the provided
        historical data.
        :param row: (pd.Series) Row of the resampled backtest dataframe with basic
                                option-related data i.e strike prices
        :param tau: (float) Maturity parameter for Black-Scholes option pricing.
        :param historical_vol_data: (dict) Historical volatility database.
        :return: (pd.Series) Row of the resampled backtest dataframe with basic
                             with calculated iv included.
        """
        datetime_key = int(row.position_open_time.timestamp())

        strike = row['call_strike']

        surface = historical_vol_data[datetime_key]
        implied_vol = surface_interpolation(iv_surface=surface, tau=tau, strike=strike)

        row['sigma_open'] = implied_vol

        output = row

        return output

    def get_resample_dict(self):
        """
        Establishes the dictionary that determines the values picked for the resampling.
        :return:
        """

        output = {'position_open_time': 'first',
                  'position_close_time': 'first',
                  'open': 'first',
                  'close': 'last',
                  'high': 'max',
                  'low': 'min',
                  'position': 'first',
                  'rv_start': 'first',
                  'rv_end': 'last'}

        if 'vol' in self.raw_data.columns:
            output['vol'] = 'last'

        if self.dvol_data_flag:
            output['dvol_open'] = 'first'
            output['dvol_close'] = 'last'
            output['dvol_premium'] = 'first'

        return output

    def get_transaction_cost(self):
        """
        Returns the transaction cost calculated for the specific strategy.
        :return: (float) Transaction cost coefficient.
        """
        output = 0

        return output

    def get_strike_prices(self, resampled_data: pd.DataFrame, strike_rounding: bool):
        """
        Calculates the option strike prices based on the resampled price data.
        :param resampled_data: (pd.DataFrame)
        :param strike_rounding: ()
        :return:
        """
        output = resampled_data.copy()
        strikes = lambda flag: output.open * (1 + flag * self.strike_otm)

        output['call_strike'] = strikes(1)
        output['put_strike'] = strikes(-1)

        if strike_rounding:
            # Returning 1 if strike price greater or equal than 1000, else 0
            # We add the 1e-6 to avoid the division by zero
            geq_thousand = lambda strike: round((strike // 1000) / (strike // 1000 + 1e-6), 0)

            # We use geq_thousand as a power of the multiplier which allows to switch between rounding
            call_rounding = 50 * 2 ** output.call_strike.apply(geq_thousand)
            put_rounding = 50 * 2 ** geq_thousand(output.put_strike)

            output.call_strike = ((output.call_strike / call_rounding).round(decimals=0) * call_rounding)

            output.put_strike = ((output.put_strike
                                  / (50 * 2 ** geq_thousand(output.put_strike))).round(decimals=0) * put_rounding)

        return output

    def get_option_parameters(self, resampled_data: pd.DataFrame, strike_rounding: bool):
        """
        Calculates parameters required for option strategy output calculation.
        :param resampled_data:
        :return:
        """
        tau = self.trade_freq / 365

        output = self.get_strike_prices(resampled_data, strike_rounding)

        if self.historical_vol_flag:
            get_implied_vol = lambda row: self.get_implied_vol(row, tau, self.historical_vol_data)

            # Adding volatility params
            output = output.apply(get_implied_vol, axis=1)

        if 'vol' in output.columns:
            output['sigma_open'] = output.vol.shift(fill_value=output.vol.iloc[0]) / 100
            output['sigma_close'] = output.vol / 100

        # Adding option prices

        output['call_price'] = black_scholes_price(output.open, output.call_strike,
                                                   tau, output.sigma_open,
                                                   self.option_rate, is_call=True) / output.open
        output['put_price'] = black_scholes_price(output.open, output.put_strike,
                                                  tau, output.sigma_open,
                                                  self.option_rate, is_call=False) / output.open

        # Adding option payoffs
        output['call_payoff'] = np.where(output.close > output.call_strike,
                                         output.close - output.call_strike, 0)

        output['put_payoff'] = np.where(output.put_strike > output.close,
                                        output.put_strike - output.close, 0)

        return output

    def get_volatility_bounds(self, volatility_data: pd.DataFrame):
        """
        {'rv': [None, None], 'dvol': [None, None], 'sigma': [None, None], 'dvol_premium': [None, None]}
        :param volatility_data:
        :return:
        """

        output = (volatility_data.sigma_open != volatility_data.sigma_open)

        for key in self.volatility_bounds:

            condition_lower = (volatility_data.sigma_open != volatility_data.sigma_open)
            condition_upper = (volatility_data.sigma_open != volatility_data.sigma_open)

            lower_bound, upper_bound = self.volatility_bounds[key]

            if key in volatility_data.columns:
                column_name = key
            else:
                raise ValueError('Incorrect bounds key value')

            if lower_bound is not None:
                condition_lower = (lower_bound > volatility_data[column_name])

            if upper_bound is not None:
                condition_upper = (volatility_data[column_name] > upper_bound)

            output = output | condition_lower | condition_upper

        return output

    @staticmethod
    def get_option_position_data(row):
        """
        Traverses pandas dataframe to
        :param row:
        :return:
        """

        if row.position == 1:
            row['option_price'] = row.call_price
            row['option_payoff'] = row.call_payoff

        elif row.position == -1:
            row['option_price'] = row.put_price
            row['option_payoff'] = row.put_payoff

        else:
            row['option_price'] = 0
            row['option_payoff'] = 0

        output = row

        return output

    def init_option_buy_position_data(self, backtest_data: pd.DataFrame):
        """
        Initializes the generalised purchased option data for both call and put positions.
        :param backtest_data: (pd.DataFrame) Dataframe with option-specific data included.
        :return: (pd.DataFrame) Dataframe with generalized option position data: price and payoff.
        """
        data = backtest_data.copy()

        position_data = pd.DataFrame(columns=['position',
                                              'call_price', 'call_payoff',
                                              'put_price', 'put_payoff',
                                              'option_price', 'option_payoff'], index=data.index)

        position_data[['position',
                       'call_price', 'call_payoff',
                       'put_price', 'put_payoff']] = data[['position',
                                                           'call_price', 'call_payoff',
                                                           'put_price', 'put_payoff']]

        position_data = position_data.apply(self.get_option_position_data, axis=1)

        #
        if self.is_usd:
            position_data.option_price = position_data.option_price * data.open
        else:
            position_data.option_payoff = position_data.option_payoff / data.close

        position_data['position_return'] = position_data.option_payoff / position_data.option_price - 1

        volatility_data = data[['rv_start', 'dvol_open', 'sigma_open', 'dvol_premium']]

        volatility_flag = self.get_volatility_bounds(volatility_data=volatility_data)

        position_data.loc[(position_data.position == 0) | volatility_flag, 'position_return'] = 0

        output = position_data

        return output

    @staticmethod
    def init_backtest_dataframe(position_return: pd.DataFrame):
        """
        Initialized the initial dataframe parameters and values before the actual backtest is performed.
        :param position_return: (pd.Series) Series of the returns of our position.
        :param backtest_data: (pd.DataFrame) Backtest initial data: OHLC and other basic parameters.
        :return: (pd.DataFrame) Initialized dataframe for the further backtest procedure.
        """

        index = position_return.index

        columns = ['option_price', 'option_payoff', 'position_return',
                   'vault_balance_start', 'vault_balance_end',
                   'invested_yield', 'trade_profit',
                   'benchmark', 'transaction_fee']

        first_row_init = [0, 0, 0,  # 'option_price', 'option_payoff', 'position_return'
                          1, 1,  # 'vault_balance_start', 'vault_balance_end'
                          0, 0,  # 'invested_yield', 'trade_profit'
                          1, 0]  # 'benchmark', 'transaction_fee'

        columns_to_specify = ['option_price', 'option_payoff', 'position_return']
        data_to_specify = position_return[['option_price', 'option_payoff', 'position_return']]

        output = pd.DataFrame(index=index, columns=columns)

        output.iloc[0] = first_row_init

        output[columns_to_specify] = data_to_specify

        return output

    def get_general_backtest_results(self, backtest_results: pd.DataFrame):
        """
        Gets the backtest result where no asset conversion is needed.

        :param backtest_results: (pd.DataFrame) Initial backtest data.
        :return: (pd.DataFrame) Backtest data and vault status records for
        the whole duration of teh backtest.
        """

        output = backtest_results.copy()

        numpy_backtest_data = output.values

        # Calculating the actual profit and loss through numpy inner values for speedup
        for i, row in enumerate(numpy_backtest_data[1:]):

            vault_balance_start_prev = numpy_backtest_data[i, 3]
            vault_balance_end_prev = numpy_backtest_data[i, 4]
            benchmark_prev = numpy_backtest_data[i, 7]

            option_price = row[0]

            option_payoff = row[1]

            position_return = row[2]

            # Start balance is the same as end balance of the prev week
            vault_balance_start = vault_balance_end_prev

            # Current invested yield in base currency is based on start balance of
            # the prev period times current rate
            invested_yield = vault_balance_start_prev * self.rate

            # To calculate the trade profit we multiply the invested yield by (position return - transaction costs)
            # If no trade is happening then no additional costs or conversions are incurred
            transaction_cost_coef = self.transaction_cost * 2

            if position_return != 0:
                transaction_cost = transaction_cost_coef * invested_yield
                trade_profit = invested_yield * position_return - transaction_cost
            else:
                transaction_cost = 0
                trade_profit = invested_yield * position_return

            # End balance is the sum of trading profit and starting balance
            vault_balance_end = vault_balance_start + invested_yield + trade_profit

            # Update benchmark balance, previous weeks benchmark plus interest
            benchmark = benchmark_prev * (1 + self.rate)

            numpy_backtest_data[i + 1] = [option_price, option_payoff, position_return,
                                          vault_balance_start, vault_balance_end,
                                          invested_yield, trade_profit,
                                          benchmark, transaction_cost]

        output.loc[:, :] = numpy_backtest_data

        return output

    def get_backtest_results(self):
        """
        Runs the main backtest procedure. This function aggregates all of the backtest
        steps and returns one joint dataframe with all of the required data.
        :return: (pd.DataFrame) Dataframe that combines all of the backtest data.
        """
        resampled_data = self.resample_by_trade_freq()

        backtest_data = self.get_option_parameters(resampled_data, self.strike_rounding)

        position_data = self.init_option_buy_position_data(backtest_data=backtest_data)

        init_dataframe = self.init_backtest_dataframe(position_return=position_data)

        output = self.get_general_backtest_results(init_dataframe)

        output = pd.concat([backtest_data, output], axis=1)

        return output

    def get_strategy_metrics(self):
        """
        Aggregates all the strategy metrics that need to be calculated for the strategy into
        a single dataframe.
        :return: (pd.DataFrame) Dataframe with strategy metrics calculated based on the backtest.
        """
        vault_ret = self.backtest_results.vault_balance_end.pct_change(1).dropna()
        strategy_apy = self.get_return_metrics('vault_balance_end')[0]
        benchmark_apy = self.get_benchmark_metrics()[0]
        metrics = {'Strategy Yield': self.get_total_yield(vault_ret),
                   'Strategy APY': strategy_apy,
                   'Benchmark APY': benchmark_apy,
                   'Trading Boost': strategy_apy / benchmark_apy,
                   'Sharpe Ratio': self.get_sharpe(vault_ret),
                   'Max Drawdown': self.get_max_drawdown(vault_ret)
                   }
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

        return metrics_df

    def get_signal_metrics(self):
        """
        Aggregates all the signal metrics that need to be calculated for the strategy into
        a single dataframe.
        :return: (pd.DataFrame) Dataframe with strategy metrics calculated based on the backtest.
        """
        longs, shorts, _ = self.get_position_distr()
        num_epochs = self.get_epochs()

        metrics = {'Epochs': num_epochs,
                   '% Trades': (longs + shorts) / num_epochs,
                   '% Longs': longs / (longs + shorts),
                   '% Shorts': shorts / (longs + shorts),
                   'Trade Hit Rate %': self.get_profitable_trades() / (longs + shorts),
                   'Signal Hit Rate %': self.get_hit_rate() / (longs + shorts)
                   }

        signal_metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

        return signal_metrics_df