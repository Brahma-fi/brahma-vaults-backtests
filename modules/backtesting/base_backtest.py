"""
Temp file for the base backtest rework
"""
from abc import ABC, abstractmethod
from datetime import timedelta
import numpy as np
import pandas as pd


class BaseBacktest(ABC):
    """
    Abstract class for further backtest calculations
    """

    def __init__(self, price_data: pd.DataFrame, position_data: pd.DataFrame,
                 trade_freq: int = 7):
        """
        Class constructor. Initializes the template for all backtests.

        :param price_data: (pd.DataFrame) OHLC high frequency data.
        :param position_data: (pd.Series) Position data in the same frequency as OHLC.
        :param trade_freq: (int) Trading frequency in days.
        """
        # Initialize the signal and price data backtest parameters
        self.raw_data = self.combine_raw_data(price_data=price_data,
                                              position_data=position_data)

        # Initialize parameters associated with the frequency of the trade
        self.trade_freq = trade_freq
        self.trades_p_year = int(365 / trade_freq)
        self.backtest_results = None
        self.transaction_cost = 0

    @staticmethod
    def combine_raw_data(price_data: pd.DataFrame, position_data: pd.DataFrame):
        """
        Combines price and position data for the raw provided dataframes.

        :param price_data:
        :param position_data:
        :return:
        """
        output = price_data.copy()

        # Add position data to OHLC
        output['position'] = position_data.fillna(0)

        return output

    @abstractmethod
    def get_resample_dict(self):
        pass

    def resample_by_trade_freq(self):
        """
        Resamples the raw data by provided trade frequency.

        :return: (pd.DataFrame) The resampled OHLC, positionand price change data in a one dataframe.
        """

        raw_data = self.raw_data.copy()

        if raw_data.index[0].hour != 8:
            raise ValueError('Correct your OHLC data to start at 8:00.')

        raw_data['position_open_time'] = raw_data.index
        raw_data['position_close_time'] = raw_data.index + timedelta(days=self.trade_freq)

        str_trade_freq = str(self.trade_freq) + 'd'

        resample_dict = self.get_resample_dict()

        output = raw_data.resample(str_trade_freq,
                                   offset='8h',
                                   label='left').agg(resample_dict)

        output['price_change'] = output.close / output.open - 1
        output.set_index('position_close_time', drop=False, inplace=True)
        output.index.name = 'time'
        output.drop(index=output.index[-1], axis=0, inplace=True)

        return output

    @abstractmethod
    def get_transaction_cost(self):
        """
        Returns the transaction cost calculated for the specific strategy.

        :return: Transaction cost coefficient.
        """
        pass

    @abstractmethod
    def get_backtest_results(self):
        """
        Runs the main backtest procedure. This function aggregates all of the backtest
        steps and returns one joint dataframe with all of the required data.

        :return: (pd.DataFrame) Dataframe that comines all of the backtest data.
        """
        pass

    @abstractmethod
    def get_strategy_metrics(self):
        """
        Aggregates all the strategy metrics that need to be calculated for the strategy into
        a single dataframe.

        :return: (pd.DataFrame) Dataframe with strategy metrics calculated based on the backtest.
        """
        pass

    @abstractmethod
    def get_signal_metrics(self):
        """
        Aggregates all the signal metrics that need to be calculated for the strategy into
        a single dataframe.

        :return: (pd.DataFrame) Dataframe with strategy metrics calculated based on the backtest.
        """
        pass

    @staticmethod
    def get_total_yield(period_ret: pd.Series) -> float:
        """
        Calculates the total yield of the strategy.

        :param period_ret: (pd.Series) A series of returns of the strategy.
        :return: The total yield obtained at the end of the backtest.
        """
        output = ((period_ret + 1).cumprod()[-1] - 1) * 100

        return output

    def get_return_metrics(self, evaluated_column: str) -> float:
        """
        Calculates the total APY and APR of the strategy.

        :param evaluated_column: (str) Name of the column we base our APY & APR calculation on.
        :return: (list) APY and APR in %
        """
        epochs = self.get_epochs()
        periods = (epochs - 1)
        tau = periods / self.trades_p_year
        apy = (self.backtest_results[evaluated_column].iloc[-1] - 1) / tau

        apr = ((self.backtest_results[evaluated_column].iloc[-1]) ** (1 / periods) - 1) * self.trades_p_year

        output = [apy * 100, apr * 100]

        return output

    def get_benchmark_metrics(self) -> float:
        """
        Calculating the benchmark APY and APR.

        :return: (list) benchmark APY and APR in %
        """
        epochs = self.get_epochs()
        periods = (epochs - 1)
        tau = (epochs - 1) / self.trades_p_year

        benchmark_apy = (self.backtest_results.benchmark.iloc[-1] - 1) / tau

        benchmark_apr = ((self.backtest_results.benchmark.iloc[-1]) ** (1 / periods) - 1) * self.trades_p_year

        output = [benchmark_apy * 100, benchmark_apr * 100]

        return output

    def get_sharpe(self, period_ret: pd.Series) -> float:
        """
        Calculating the Sharpe ratio based on the provided weekly returns series.

        :param period_ret: (pd.Series) Time series of weekly returns.
        :returns: (float) The Sharpe ratio of the strategy.
        """

        weekly_sharpe = period_ret.mean() / period_ret.std() if period_ret.std() != 0 else period_ret.mean() / 10000

        annualized_sharpe = np.sqrt(self.trades_p_year) * weekly_sharpe

        return annualized_sharpe

    def get_max_drawdown(self, period_ret: pd.Series):
        """
        Calculating the maximum drawdown of the strategy for the provided weekly returns series.

        :param period_ret: (pd.Series) Time series of weekly returns.
        :returns: (float) The maximum drawdown for the strategy.
        """
        period_ret_cum = (period_ret + 1).cumprod()

        window = self.trades_p_year
        rolling_max = period_ret_cum.rolling(window, min_periods=1).max()

        weekly_drawdown = period_ret_cum / rolling_max - 1

        max_drawdown = weekly_drawdown.rolling(window, min_periods=1).min()

        return max_drawdown[-1]

    def get_sortino(self, period_ret: pd.Series) -> float:
        """
        Calculating the Sortino ratio based on the provided weekly returns series.

        :param period_ret: (pd.Series) Time series of weekly returns.
        :returns: (float) The Sortino ratio of the strategy.
        """
        if period_ret[period_ret < 0].std() != 0:
            std = period_ret[period_ret < 0].std()
        else:
            std = 0.00001

        weekly_sortino = period_ret.mean() / std

        annualized_sortino = np.sqrt(self.trades_p_year) * weekly_sortino

        return annualized_sortino

    def get_calmar(self, period_ret: pd.Series) -> float:
        """
        Calculating the Calmar ratio based on the provided weekly returns series.

        :param period_ret: (pd.Series) Time series of weekly returns.
        :returns: (float) The Calmar ratio of the strategy.
        """
        calmar = period_ret.mean() * self.trades_p_year / abs(self.get_max_drawdown(period_ret))

        return calmar

    def get_epochs(self):
        """
        Calculating the number of trading epochs in the backtest.

        :return: Number of epochs.
        """
        num_epochs = self.backtest_results.shape[0]

        return num_epochs

    def get_position_distr(self):
        """
        Calculating the position distribution between longs, shorts and no_trades

        :return:
        """
        longs = (self.backtest_results.position == 1).sum()
        shorts = (self.backtest_results.position == -1).sum()
        no_trade = (self.backtest_results.position == 0).sum()

        output = [longs, shorts, no_trade]

        return output

    def get_profitable_trades(self):
        """
        Calculates the amount of profitable trades.

        :return: (int) Amount of profitable trades.
        """
        profitable_trades = (self.backtest_results.position_return > 0).sum()

        return profitable_trades

    def get_hit_rate(self):
        """
        Calculates the signal correct hit rate.

        :return: (int) Signal correct hit rate.
        """
        signal_correct = (self.backtest_results.position * self.backtest_results.price_change > 0).sum()

        return signal_correct
