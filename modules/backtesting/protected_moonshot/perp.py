import numpy as np


def run_perp_backtest(weekly_inputs, hourly_inputs, freq, leverage, mm_min, interest,
                      currency, strategy, take_profit=0.15, stop_loss=0.05):
    # weeklyInputData Pandas dataframe, weekly frequency
    # Columns required, open,close,high,low,price_change,position
    output_data = weekly_inputs.copy()

    short_liq = (1 + leverage) / (leverage * (mm_min + 1)) - 1
    long_liq = 1 - (1 - leverage) / (leverage * (mm_min - 1))

    if short_liq < stop_loss or long_liq < stop_loss:
        print("warning stop loss to low for leverage")
        stop_loss = min(short_liq, long_liq) / 1.1

    if strategy == 'simple':
        output_data['liq_price'] = np.where(weekly_inputs.position > 0, weekly_inputs.open * (1 - long_liq),
                                           weekly_inputs.open * (1 + short_liq))
        output_data['perp_returns'] = 0

        mask_long_liqs = ((weekly_inputs.position > 0) & (weekly_inputs.low < output_data.liq_price))
        mask_short_liqs = ((weekly_inputs.position < 0) & (weekly_inputs.high > output_data.liq_price))
        mask_else = (mask_long_liqs == False) & (mask_short_liqs == False)

        if currency == 'USD':
            output_data.loc[(mask_long_liqs | mask_short_liqs), 'perp_returns'] = -interest
            output_data.loc[mask_else, 'perp_returns'] = interest * leverage * weekly_inputs.loc[mask_else, 'position'] * \
                                                      weekly_inputs.loc[mask_else, 'price_change']
        elif currency == 'ETH':
            output_data.loc[(mask_long_liqs | mask_short_liqs), 'perp_returns'] = -interest

            margin_return = interest * (weekly_inputs.loc[mask_else, 'open'] / weekly_inputs.loc[mask_else, 'close'] - 1)
            trading_profit_eth = interest * leverage * weekly_inputs.loc[mask_else, 'position'] * (
                    1 - weekly_inputs.loc[mask_else, 'open'] / weekly_inputs.loc[mask_else, 'close'])
            output_data.loc[mask_else, 'perp_returns'] = margin_return + trading_profit_eth

    elif strategy == 'enhanced':
        # enhanced perp strategy returns
        output_data['take_profit'] = np.where(weekly_inputs.position > 0, weekly_inputs.open * (1 + take_profit),
                                             weekly_inputs.open * (1 - take_profit))
        output_data['stop_loss'] = np.where(weekly_inputs.position > 0, weekly_inputs.open * (1 - stop_loss),
                                           weekly_inputs.open * (1 + stop_loss))

        # find exit flag
        output_data['exit_flag'] = 0
        exit_flags = check_perp_exit(output_data, hourly_inputs, take_profit, stop_loss, long_liq, short_liq, freq)
        output_data.loc[output_data.index, 'exit_flag'] = exit_flags.copy()

        mask_take_profit = (output_data.exit_flag == 'take_profit')
        mask_stop_loss = (output_data.exit_flag == 'stop_loss')
        mask_in_range = (output_data.exit_flag == 'in_range')

        if currency == 'USD':
            output_data.loc[mask_take_profit, 'perp_returns'] = interest * leverage * take_profit
            output_data.loc[mask_stop_loss, 'perp_returns'] = -interest * leverage * stop_loss
            output_data.loc[mask_in_range, 'perp_returns'] = interest * leverage * weekly_inputs.loc[
                mask_in_range, 'position'] * weekly_inputs.loc[mask_in_range, 'price_change']
        elif currency == 'ETH':
            margin_return = interest * (weekly_inputs['open'] / weekly_inputs['close'] - 1)

            output_data.loc[mask_take_profit, 'perp_returns'] = interest * leverage * take_profit * weekly_inputs.loc[
                mask_take_profit, 'open'] / weekly_inputs.loc[mask_take_profit, 'close'] + margin_return.loc[mask_take_profit]
            output_data.loc[mask_stop_loss, 'perp_returns'] = -interest * leverage * stop_loss * weekly_inputs.loc[
                mask_stop_loss, 'open'] / weekly_inputs.loc[mask_stop_loss, 'close'] + margin_return.loc[mask_stop_loss]
            output_data.loc[mask_in_range, 'perp_returns'] = interest * leverage * weekly_inputs.loc[
                mask_in_range, 'position'] * (1 - weekly_inputs.loc[mask_in_range, 'open'] / weekly_inputs.loc[
                mask_in_range, 'close']) + margin_return.loc[mask_in_range]

    else:
        print("Incorrect strategy name")

    alpha = output_data.perp_returns.sum() / (output_data.shape[0] / 52)

    return output_data, alpha


def check_perp_exit(exit_data, hourly_data, take_profit, stop_loss, long_liq, short_liq, freq):
    for i, row in exit_data.iloc[0:, :].iterrows():
        flag = 0
        loc = hourly_data.index.get_loc(i)
        weeks_prices = hourly_data.close.iloc[loc - freq * 24:loc]

        if row.position < 0:

            take_profit_count = np.sum(weeks_prices < row.open * (1 - take_profit), axis=0)
            stop_loss_count = np.sum(weeks_prices > row.open * (1 + stop_loss), axis=0)

            if take_profit_count + stop_loss_count == 0:
                flag = 'in_range'
            elif (take_profit_count + stop_loss_count) == take_profit_count:
                flag = 'take_profit'
                exit_price = row.open * (1 - take_profit)
            elif (take_profit_count + stop_loss_count) == stop_loss_count:
                flag = 'stop_loss'
                exit_price = row.open * (1 + stop_loss)
            else:  # stop loss and take profit hit
                tp = (weeks_prices < row.take_profit).argmax(axis=0)
                sl = (weeks_prices > row.stop_loss).argmax(axis=0)

                if tp < sl:
                    flag = 'take_profit'
                    exit_price = row.open * (1 - take_profit)
                else:
                    flag = 'stop_loss'
                    exit_price = row.open * (1 + stop_loss)


        elif row.position > 0:

            take_profit_count = np.sum(weeks_prices > row.open * (1 + take_profit), axis=0)
            stop_loss_count = np.sum(weeks_prices < row.open * (1 - stop_loss), axis=0)

            if take_profit_count + stop_loss_count == 0:
                flag = 'in_range'
            elif (take_profit_count + stop_loss_count) == take_profit_count:
                flag = 'take_profit'
                exit_price = row.open * (1 + take_profit)
            elif (take_profit_count + stop_loss_count) == stop_loss_count:
                flag = 'stop_loss'
                exit_price = row.open * (1 - stop_loss)
            else:  # stop loss and take profit hit
                tp = (weeks_prices > row.take_profit).argmax(axis=0)
                sl = (weeks_prices < row.stop_loss).argmax(axis=0)

                if tp < sl:
                    flag = 'take_profit'
                    exit_price = row.open * (1 + take_profit)
                else:
                    flag = 'stop_loss'
                    exit_price = row.open * (1 - stop_loss)

        exit_data.loc[i, 'exit_flag'] = flag

    return exit_data['exit_flag']