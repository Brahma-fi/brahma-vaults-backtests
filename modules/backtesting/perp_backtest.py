"""
Perp Backtest module.
"""
from warnings import warn
import numpy as np
import pandas as pd

from modules.backtesting.base_backtest import BaseBacktest

class PerpBacktest(BaseBacktest):
    """
    Backtest for the vault-based strategy trading perpetual futures.
    """

    def __init__(self, price_data: pd.DataFrame, position_data: pd.DataFrame,
                 rate: float, trade_freq: int = 7, leverage: int = None,
                 take_profit: float = None, stop_loss: float = None,
                 is_usd: bool = True):
        """
        Class constructor. Also initializes the whole backtesting process in one run.

        :param price_data: (pd.DataFrame) Hourly OHLC price data.
        :param position_data: (pd.Series) Hourly position data.
        :param rate: (float) Compounding rate for the vault.
        :param trade_freq: (int) Trading frequency in days: 7, 3, etc.
        :param take_profit: (float) Take profit bound.
        :param stop_loss: (float) Stop-loss bound.
        :param leverage: (int) The amount of leverage used for trades.
        :param is_usd: (bool) Are the trades made in one currency or converted between multiple.
        """
        # Initialize basic backtest parameters
        super().__init__(price_data=price_data, position_data=position_data,
                         trade_freq=trade_freq)

        self.margin = 0.0625
        self.rate = rate * self.trade_freq / 365
        self.leverage = leverage
        self.liquidation_bounds = self.get_liquidation_bounds()
        self.pnl_bounds = self.get_pnl_bounds(take_profit, stop_loss)
        self.is_usd = is_usd
        self.transaction_cost = self.get_transaction_cost()

        # Get backtest results
        self.backtest_results = self.get_backtest_results()
        self.strategy_metrics = self.get_strategy_metrics()
        self.signal_metrics = self.get_signal_metrics()

    def get_resample_dict(self):
        """
        Prepares a resample aggregator dictionary.
        :return: (dict) A dictionary that is outlining the aggregation functions for each raw
                        data column.
        """

        output = {'position_open_time': 'first',
                  'position_close_time': 'first',
                  'open': 'first',
                  'close': 'last',
                  'high': 'max',
                  'low': 'min',
                  'position': 'first'}

        return output

    def get_liquidation_bounds(self):
        """
        Sets the liquidation bounds for margin trade. Depends on the exchange used.

        :return: (list) Liquidation bounds for a given leverage level and platform.
        """
        output = None

        if self.leverage is not None:
            short_liquidation = (((1 + self.leverage)
                                  / (self.leverage * (self.margin + 1))) - 1)

            long_liquidation = (1 - ((1 - self.leverage)
                                     / (self.leverage * (self.margin - 1))))

            output = [short_liquidation, long_liquidation]

        return output

    def get_pnl_bounds(self, take_profit: float, stop_loss: float):
        """
        Sets the take-profit and stop-loss value for current strategy. The function accounts
        for the case when given stop-loss is not enough to avoid the liquidation case
        providing the optimal value closest to given param.

        :param take_profit: (float) Take profit value.
        :param stop_loss: (float) Initial stop-loss value.
        :return: (list) List of take profit and stop loss value.
        """
        if stop_loss is not None:

            if self.liquidation_bounds[0] < stop_loss or self.liquidation_bounds[1] < stop_loss:
                output = min(self.liquidation_bounds) / 1.1

                # Warn users about the new set value
                warn(f'Provided stop loss is too high. Adjusted stop loss value is: {output}',
                     UserWarning)

        output = [stop_loss, take_profit]

        return output

    def get_transaction_cost(self):
        """
        Returns the transaction cost calculated for the specific strategy.

        :return: Transaction cost coefficient.
        """

        output = 1e-3

        return output

    def get_exit_event(self, row: pd.Series):
        """
        Calculates whether an exit event had happened during trading epoch.
        The function adds position duration and type of exit (-1 - sl, 1 - tp, 0 - none).


        :param row: (pd.Series) Row of the backtest dataframe
        :return: (pd.Series) Row of the backtest dataframe including position exit data.
        """

        end = row.position_close_time
        start = row.position_open_time

        flag = 0

        if row.position == 0:
            end = start
            row['exit_type'] = flag
            row['trade_duration'] = end - start
            output = row
            return output

            # Extract the hourly epoch prices from the raw data
        epoch_prices = self.raw_data.open.loc[row.position_open_time:row.position_close_time]

        # Get the stop loss/take profit occurrence index based on the epoch prices
        if row.position == 1:

            take_profit_hit = epoch_prices.gt(row.take_profit_price).idxmax()
            stop_loss_hit = epoch_prices.lt(row.stop_loss_price).idxmax()

        elif row.position == -1:

            take_profit_hit = epoch_prices.lt(row.take_profit_price).idxmax()
            stop_loss_hit = epoch_prices.gt(row.stop_loss_price).idxmax()



        both_exits_hit = [x != start for x in [take_profit_hit, stop_loss_hit]]

        # Checking for the first entry for flagging
        if all(both_exits_hit):
            # If both events took place within the epoch set the flag corresponding to the
            # earliest one encountered
            flag = 1 if take_profit_hit < stop_loss_hit else -1

        elif take_profit_hit != stop_loss_hit:
            # Otherwise set the flag assuming the unoccurred hit assumed the 'start' value
            flag = 1 if take_profit_hit > stop_loss_hit else -1

        else:

            flag = 0

        # Calculating the duration of the trade based on the exit timing
        if flag == 1:
            end = take_profit_hit
        elif flag == -1:
            end = stop_loss_hit
        else:
            pass

        trade_duration = end - start

        row['exit_type'] = flag
        row['trade_duration'] = trade_duration

        output = row

        return output

    def get_exit_flags(self, resampled_data: pd.DataFrame):
        """
        General function responsible for establishing exit events for the whole backtest period.

        :param resampled_data:
        :return:
        """
        position_flag = (resampled_data.position != 0).astype(int)

        resampled_data['stop_loss_price'] = resampled_data.open * (
                    1 - resampled_data.position * self.pnl_bounds[0]) * position_flag
        resampled_data['take_profit_price'] = resampled_data.open * (
                    1 + resampled_data.position * self.pnl_bounds[1]) * position_flag

        resampled_data = resampled_data.apply(self.get_exit_event, axis=1)

        return resampled_data

    def init_backtest_dataframe(self, position_return: pd.DataFrame, backtest_data: pd.DataFrame):
        """
        Initialized the initial dataframe parameters and values before the actual backtest is performed.

        :param position_return: (pd.Series) Series of the returns of our position.
        :param backtest_data: (pd.DataFrame) Backtest initial data: OHLC and other basoic parameters.
        :return: (pd.DataFrame) Initialized dataframe for the further backtest procedure.
        """

        index = position_return.index

        if self.is_usd:

            columns = ['position_return',
                       'vault_balance_start', 'vault_balance_end',
                       'invested_yield', 'trade_profit',
                       'benchmark', 'transaction_fee']

            first_row_init = [0,  # 'position_return'
                              1, 1,  # 'vault_balance_start', 'vault_balance_end'
                              0, 0,  # 'invested_yield', 'trade_profit'
                              1, 0]  # 'benchmark', 'transaction_fee'

            columns_to_specify = ['position_return']
            data_to_specify = position_return

        else:

            columns = ['position_return',
                       'vault_balance_start', 'vault_balance_end',
                       'invested_yield_base', 'invested_yield_usd',
                       'trade_profit_base',
                       'open_price', 'close_price',
                       'benchmark', 'transaction_fee']

            first_row_init = [0,  # 'position_return'
                              1, 1,  # 'vault_balance_start', 'vault_balance_end'
                              0, 0,  # 'invested_yield_base', 'invested_yield_usd'
                              0,  # 'trade_profit_base'
                              0, 0,  # 'open_price', 'close_price'
                              1, 0]  # 'benchmark', 'transaction_fee'

            columns_to_specify = ['position_return', 'open_price', 'close_price']
            data_to_specify = pd.concat([position_return,
                                         backtest_data.open[position_return.index],
                                         backtest_data.close[position_return.index]], axis=1)

        output = pd.DataFrame(index=index, columns=columns)

        output.iloc[0] = first_row_init

        output[columns_to_specify] = data_to_specify

        return output

    def init_liquidation_bound_backtest(self, backtest_data: pd.DataFrame):
        """
        Initializes for the backtest where our position can be liquidated.

        :param backtest_data: (pd.DataFrame) Backtest data.
        :return: (pd.DataFrame) Initial backtest data including liquidations data and position returns.
        """
        # Setting liq. price where is long position then liquidation price is 1 - liq. amt, otherwise its 1 + liq
        backtest_data['liquidation_price'] = np.where(backtest_data.position > 0,
                                                      backtest_data.open * (1 - self.liq_bounds[1]),
                                                      backtest_data.open * (1 + self.liq_bounds[0]))

        backtest_data['liquidation_price'][backtest_data.position == 0] = 0

        # Getting the indexes from the liquidation events
        long_liqs = ((backtest_data.position > 0) & (backtest_data.low < backtest_data.liquidation_price))
        short_liqs = (backtest_data.position < 0) & (backtest_data.high > backtest_data.liquidation_price)

        # Setting the raw signal-based position returns
        position_return = backtest_data.position * backtest_data.price_change * self.leverage

        # We lose the whole amt at liquidation
        position_return.loc[long_liqs | short_liqs] = -1

        output = [position_return, backtest_data]

        return output

    def init_pnl_bound_backtest(self, backtest_data: pd.DataFrame):
        """
        Initialization for the backtest where our position is bound by stop-loss and take profit..

        :param backtest_data: (pd.DataFrame) Backtest data.
        :return: (pd.DataFrame) Initial backtest data including exit events data and position returns.
        """

        backtest_data_with_exit = self.get_exit_flags(backtest_data)

        # Getting the timings for all of the pnl bound events and the duration of the trades
        position_return = backtest_data_with_exit.position * backtest_data_with_exit.price_change * self.leverage

        # We don't lose the whole amt at liquidation
        position_return.loc[backtest_data_with_exit.exit_type == -1] = -self.pnl_bounds[0] * self.leverage
        position_return.loc[backtest_data_with_exit.exit_type == 1] = self.pnl_bounds[1] * self.leverage
        position_return.loc[backtest_data_with_exit.position == 0] = 0

        output = [position_return, backtest_data_with_exit]

        return output

    def get_one_currency_backtest_results(self, backtest_results: pd.DataFrame):
        """
        Gets the backtest result where no asset conversion is needed.

        :param backtest_results: (pd.DataFrame) Inintial backtest data.
        :return: (pd.DataFrame) Backtest data and vault status records for
        the whole duration of teh backtest.
        """

        output = backtest_results.copy()

        numpy_backtest_data = output.values

        # Calculating the actual profit and loss through numpy inner values for speedup
        for i, row in enumerate(numpy_backtest_data[1:]):

            vault_balance_start_prev = numpy_backtest_data[i, 1]
            vault_balance_end_prev = numpy_backtest_data[i, 2]
            benchmark_prev = numpy_backtest_data[i, 5]

            position_return = row[0]

            # Start balance is the same as end balance of the prev week
            vault_balance_start = vault_balance_end_prev

            # Current invested yield in base currency is based on start balance of
            # the prev period times current rate
            invested_yield = vault_balance_start_prev * self.rate

            # To calculate the trade profit we multiply the invested yield by (position return - transaction costs)
            # If no trade is happening then no additional costs or conversions are incurred
            transaction_cost_coef = self.leverage * self.transaction_cost * 2

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

            numpy_backtest_data[i + 1] = [position_return,
                                          vault_balance_start, vault_balance_end,
                                          invested_yield, trade_profit,
                                          benchmark, transaction_cost]

        output.loc[:, :] = numpy_backtest_data

        return output

    def get_multicurrency_backtest_results(self, backtest_results: pd.DataFrame):
        """


        :param backtest_results: (pd.DataFrame)
        :return: (pd.DataFrame)
        """

        output = backtest_results.copy()
        numpy_backtest_data = output.values

        # Calculating the actual profit and loss through numpy inner values for speedup
        for i, row in enumerate(numpy_backtest_data[1:]):

            vault_balance_start_prev = numpy_backtest_data[i, 1]
            vault_balance_end_prev = numpy_backtest_data[i, 2]
            benchmark_prev = numpy_backtest_data[i, -2]

            position_return = row[0]
            open_price = row[6]
            close_price = row[7]

            # Start balance is the same as end balance of the prev week
            vault_balance_start = vault_balance_end_prev

            # Current invested yield in base currency is based on start balance of the prev period times current rate
            invested_yield_base = vault_balance_start_prev * self.rate

            # To convert into usd we multiply it by asset price at period's open
            invested_yield_usd = invested_yield_base * open_price

            # To calculate the trade profit we multiply the invested yield by (1 + position return - transaction costs)
            # and divide by close price for the week to get the value in current asset

            # If no trade is happening then no additional costs or conversions are incurred
            transaction_cost_coef = self.leverage * self.transaction_cost * 2

            if position_return != 0:
                # The amount of regular transaction fee
                transaction_cost = invested_yield_usd * transaction_cost_coef
                # Overall pnl
                trade_profit_base = (invested_yield_usd * position_return - transaction_cost) / close_price
            else:
                trade_profit_base = invested_yield_base * position_return
                transaction_cost = 0

            # End balance is the sum of trading profit and starting balance
            vault_balance_end = vault_balance_start + invested_yield_base + trade_profit_base

            # Update benchmark balance, previous weeks benchmark plus interest
            benchmark = benchmark_prev * (1 + self.rate)

            numpy_backtest_data[i + 1] = [position_return,
                                          vault_balance_start, vault_balance_end,
                                          invested_yield_base, invested_yield_usd,
                                          trade_profit_base,
                                          open_price, close_price,
                                          benchmark, transaction_cost]

        output.loc[:, :] = numpy_backtest_data

        output = output.drop(['open_price', 'close_price'], axis=1)

        return output

    def get_backtest_results(self):
        """
        Runs the main backtest procedure. This function aggregates all of the backtest
        steps and returns one joint dataframe with all of the required data.

        :return: (pd.DataFrame) Dataframe that combines all of the backtest data.
        """

        resampled_data = self.resample_by_trade_freq()

        # Initializes the backtest returns and position returns depending on the type of bounds
        if all([x is not None for x in self.liquidation_bounds]):
            position_return, backtest_data = self.init_pnl_bound_backtest(resampled_data.copy())
        else:
            position_return, backtest_data = self.init_liquidation_bound_backtest(resampled_data.copy())

        init_dataframe = self.init_backtest_dataframe(position_return=position_return, backtest_data=backtest_data)

        if self.is_usd:
            output = self.get_one_currency_backtest_results(init_dataframe)

        else:
            output = self.get_multicurrency_backtest_results(init_dataframe)

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
        longs, shorts, no_trade = self.get_position_distr()
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
