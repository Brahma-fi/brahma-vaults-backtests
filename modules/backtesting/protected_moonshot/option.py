import numpy as np
import pandas as pd
import sys
sys.path.append('../../')
from modules.options.basic_option_helpers import black_scholes_price


def run_option_backtest(weekly_inputs, weekly_vol, freq, interest, currency,
                        r, strike_otm, strategy, strike_rounding=False):
    # weeklyInputData Pandas dataframe, weekly frequency
    # Columns required, open,close,position

    output_data = pd.DataFrame(index=weekly_inputs.index)
    output_data['optionReturns'] = 0

    tau = freq / 365

    spot_prices = weekly_inputs.open

    if strike_rounding:
        call_strikes = round(spot_prices * (1 + strike_otm) / 100.0, 0) * 100.0
        put_strikes = round(spot_prices * (1 - strike_otm) / 100.0, 0) * 100.0
    else:
        call_strikes = spot_prices * (1 + strike_otm)
        put_strikes = spot_prices * (1 - strike_otm)

    output_data['sigma_open'] = weekly_vol.close.shift(fill_value=weekly_vol.close.iloc[0]) / 100
    output_data['sigma_close'] = weekly_vol.close / 100

    output_data['call_prices'] = black_scholes_price(spot_prices, call_strikes, tau, output_data['sigma_open'],
                                                     r, 1) / spot_prices
    output_data['put_prices'] = black_scholes_price(spot_prices, put_strikes, tau, output_data['sigma_open'],
                                                    r, -1) / spot_prices

    output_data['call_payoff'] = np.where(weekly_inputs.close > call_strikes, weekly_inputs.close - call_strikes, 0)
    output_data['put_payoff'] = np.where(put_strikes > weekly_inputs.close, put_strikes - weekly_inputs.close, 0)

    if strategy == 'option_buyer':
        mask_long = (weekly_inputs.position > 0)
        mask_short = (weekly_inputs.position < 0)

        if currency == 'USD':
            output_data.loc[mask_long, 'option_returns'] = interest / (
                    output_data.loc[mask_long, 'call_prices'] * weekly_inputs.loc[mask_long, 'open']) * output_data.loc[
                                                               mask_long, 'call_payoff'] - interest
            output_data.loc[mask_short, 'option_returns'] = interest / (output_data.loc[mask_short, 'put_prices']
                                                                        * weekly_inputs.loc[mask_short, 'open']) * \
                                                            output_data.loc[mask_short, 'put_payoff'] - interest
        elif currency == 'ETH':
            output_data.loc[mask_long, 'option_returns'] = interest / output_data.loc[mask_long, 'call_prices'] * \
                                                           output_data.loc[mask_long, 'call_payoff'] / \
                                                           weekly_inputs.loc[
                                                               mask_long, 'close'] - interest
            output_data.loc[mask_short, 'option_returns'] = interest / output_data.loc[mask_short, 'put_prices'] * \
                                                            output_data.loc[mask_short, 'put_payoff'] / \
                                                            weekly_inputs.loc[
                                                                mask_short, 'close'] - interest

    elif strategy == 'straddle_buyer':
        if currency == 'USD':
            output_data['option_returns'] = (interest / 2) / (output_data['call_prices'] * weekly_inputs['open']) * \
                                            output_data['call_payoff'] + (interest / 2) / (
                                                    output_data['put_prices'] * weekly_inputs['open']) * output_data[
                                                'put_payoff'] - interest
        elif currency == 'ETH':
            calls_bought = (interest / 2) / output_data['call_prices']
            puts_bought = (interest / 2) / output_data['put_prices']

            call_payoff = calls_bought * output_data['call_payoff'] / weekly_inputs['close']
            put_payoff = puts_bought * output_data['put_payoff'] / weekly_inputs['close']
            output_data['option_returns'] = call_payoff + put_payoff - interest

    elif strategy == 'option_seller':
        mask_long = (weekly_inputs.position > 0)
        mask_short = (weekly_inputs.position < 0)

        if currency == 'USD':
            output_data.loc[mask_short, 'option_returns'] = -interest / (weekly_inputs.loc[mask_short, 'open']) * \
                                                            output_data.loc[mask_short, 'call_payoff'] + interest * \
                                                            output_data.loc[mask_short, 'call_prices']
            output_data.loc[mask_long, 'option_returns'] = -interest / (weekly_inputs.loc[mask_long, 'open']) * \
                                                           output_data.loc[mask_long, 'put_payoff'] + interest * \
                                                           output_data.loc[mask_long, 'put_prices']
        elif currency == 'ETH':
            output_data.loc[mask_short, 'option_returns'] = -interest * output_data.loc[mask_short, 'call_payoff'] / \
                                                            weekly_inputs.loc[mask_short, 'close'] + interest * \
                                                            output_data.loc[mask_short, 'call_prices']
            output_data.loc[mask_long, 'option_returns'] = -interest * output_data.loc[mask_long, 'put_payoff'] / \
                                                           weekly_inputs.loc[mask_long, 'close'] + interest * \
                                                           output_data.loc[
                                                               mask_long, 'put_prices']

    elif strategy == 'straddle_seller':
        if currency == 'USD':
            calls_sold = (interest / 2) / weekly_inputs['open']
            puts_sold = (interest / 2) / weekly_inputs['open']  # assuming atm puts
            call_premium_earned = calls_sold * (output_data['call_prices'] * weekly_inputs['open'])
            put_premium_earned = puts_sold * (output_data['put_prices'] * weekly_inputs['open'])
            call_payoff = - calls_sold * output_data['call_payoff']
            put_payoff = - puts_sold * output_data['put_payoff']
            output_data['option_returns'] = call_premium_earned + put_premium_earned + call_payoff + put_payoff
        elif currency == 'ETH':
            calls_sold = (interest / 2)
            puts_sold = (interest / 2)  # assuming atm puts
            call_premium_earned = calls_sold * output_data['call_prices']
            put_premium_earned = puts_sold * output_data['put_prices']
            call_payoff = - calls_sold * output_data['call_payoff'] / weekly_inputs['close']
            put_payoff = - puts_sold * output_data['put_payoff'] / weekly_inputs['close']
            output_data['option_returns'] = call_premium_earned + put_premium_earned + call_payoff + put_payoff
    else:
        print(strategy + " is not a known option strategy")

    alpha = output_data.option_returns.sum() / (output_data.shape[0] / 52)

    return output_data, alpha
