"""
Temp module docstring
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


# pylint: disable=invalid-name,
def option_payoff(spot, strike, flag):
    """
    Calculates the option payoff.
    :param spot:
    :param strike:
    :param flag:
    :return:
    """
    payoff = max(flag * (spot - strike), 0)

    return payoff


def black_scholes_price_old(spot: float, strike: float, tau: float,
                            sigma: float, r: float, flag: int):
    """
    Calculates the option price.
    :param spot: (float) Spot price.
    :param strike: (float) Strike price.
    :param tau: (float) Time to maturity.
    :param sigma: (float) Implied volatility.
    :param r: (float) Risk-free-rate.
    :param flag: Whether the price is calculated for call (1) or for put (-1).
    :return: (float) Option price.
    """

    d1 = (np.log(spot / strike) + (r + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    price = flag * spot * norm.cdf(flag * d1) - flag * strike * np.exp(-r * tau) * norm.cdf(flag * d2)

    return price


def black_scholes_price(spot: float, strike: float, tau: float, sigma: float, rate: float, is_call: bool):
    """
    Calculates the option price.
    :param spot: (float) Spot price.
    :param strike: (float) Strike price.
    :param tau: (float) Time to maturity.
    :param sigma: (float) Implied volatility.
    :param rate: (float) Risk-free-rate.
    :param is_call: (bool) Whether the price is calculated for call or for put option.

    :return: (float) Option price.
    """
    flag = 1 if is_call else -1

    d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    price = flag * spot * norm.cdf(flag * d1) - flag * strike * np.exp(-rate * tau) * norm.cdf(flag * d2)

    return price


def get_option_expiry_value(spot_expiry_price: float, strike: float, is_call: bool):
    """
    Calculates the option payoff at expiry.
    :param spot_expiry_price: (float) Spot price at expiry.
    :param strike: (float) Option Strike price.
    :param is_call: (bool) Whether the option is a call or put option.
    :return: (float) Option expiry price.
    """
    flag = 1 if is_call else -1

    expiry_value = max(flag*spot_expiry_price - flag*strike, 0)

    return expiry_value


def black_scholes_delta(spot: float, strike: float, tau: float, sigma: float, rate: float, is_call: bool):
    """
    Calculates the black-scholes delta of an option.
    :param spot: (float) Spot price.
    :param strike: (float) Strike price.
    :param tau: (float) Time to maturity.
    :param sigma: (float) Implied volatility.
    :param rate: (float) Risk-free-rate.
    :param is_call: (bool) Whether the price is calculated for call or for put option.
    :return: (float) Option delta.
    """
    flag = 1 if is_call else -1

    d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))

    delta = flag*norm.cdf(flag*d1)

    return delta


def black_scholes_delta_from_surface(spot: float, strike: float, tau: float,
                                     iv_surface: pd.DataFrame, rate: float, is_call: bool):
    """
    Calculates the delta of an option with IV calculated using the Implied Volatility surface.
    :param spot: (float) Spot price.
    :param strike: (float) Strike price.
    :param tau: (float) Time to maturity.
    :param iv_surface: (pd.DataFrame) Implied volatility surface with columns 'tau','strike' and 'mark_iv'
    :param rate: (float) Risk-free-rate.
    :param is_call: (bool) Whether the price is calculated for call or for put option.
    :return: (float) Option delta.
    """
    sigma = surface_interpolation(iv_surface, tau, strike)

    flag = 1 if is_call else -1

    d1 = (np.log(spot / strike) + (rate + sigma ** 2 / 2) * tau) / (sigma * np.sqrt(tau))

    delta = flag*norm.cdf(flag*d1)

    return delta


def get_risk_free_rate(spot: float, strike: float, tau: float, sigma: float,
                       option_price: float, is_call: bool):
    """
    Calculates the risk free rate from given parameters and option price.
    
    :param spot: (float) Spot price.
    :param strike: (float) Strike price.
    :param tau: (float) Time to maturity.
    :param sigma: (float) Implied volatility.
    :param option_price: (float) Option price.
    :param is_call: (bool) Whether the price is calculated for call or for put option.

    :return: (list) Derived risk-free rate and proximity to a true answer.
    """
    func = lambda r: black_scholes_price(spot=spot, strike=strike, tau=tau,
                                         sigma=sigma, rate=r, is_call=is_call) - option_price

    root = fsolve(func, [0.1])[0]

    output = [root, func(root)]

    return output


def get_mark_iv(spot: float, strike: float, tau: float, sigma: float,
                option_price: float, mark_price: float, is_call: bool):
    """
    Calculates the mark implied volatility rate from given parameters and option price.

    :param spot: (float) Spot price.
    :param strike: (float) Strike price.
    :param tau: (float) Time to maturity.
    :param sigma: (float) Implied volatility.
    :param option_price: (float) Option price.
    :param mark_price: (float) Mark option price.
    :param is_call: (bool) Whether the price is calculated for call or for put option.

    :return: (list) Derived mark implied volatility and proximity to a true answer.
    """

    rate = get_risk_free_rate(spot=spot, strike=strike, tau=tau,
                              sigma=sigma, option_price=option_price, is_call=is_call)[0]

    func = lambda mark_iv: black_scholes_price(spot=spot, strike=strike, tau=tau, sigma=mark_iv, rate=rate,
                                               is_call=is_call) - mark_price

    root = fsolve(func, [0.9 * sigma])[0]

    output = [root, func(root)]

    return output


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
    func = interp1d(x, y, bounds_error=False, fill_value=(y[0], y[-1]))

    iv = func(strike)

    return iv


def time_interpolation(near_skew, far_skew, tau, strike):
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

    near_iv = strike_interpolation(near_skew, strike)
    far_iv = strike_interpolation(far_skew, strike)

    near_tau = near_skew.tau.iloc[0]
    far_tau = far_skew.tau.iloc[0]

    if near_tau > far_tau:
        raise ValueError('Near Skew has expiry date after Far Skew, adjust inputs')

    cum_variance_near = near_iv**2 * near_tau
    cum_variance_far = far_iv**2 * far_tau

    f = interp1d([near_tau, far_tau], [cum_variance_near, cum_variance_far])

    cum_variance = f(tau)
    iv = np.sqrt(cum_variance/tau)

    return iv


def surface_interpolation(iv_surface, tau, strike):
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
        iv = strike_interpolation(skew, strike) + 0

    elif tau < unique_expiries.min():
        skew = iv_surface[iv_surface.tau == unique_expiries.min()][['strike', 'mark_iv']]
        iv = strike_interpolation(skew, strike) + 0

    elif tau > unique_expiries.max():
        skew = iv_surface[iv_surface.tau == unique_expiries.max()][['strike', 'mark_iv']]
        iv = strike_interpolation(skew, strike) + 0

    else:
        less_than_mask = unique_expiries < tau
        near_expiry_idx = np.where(less_than_mask)[0][-1]

        greater_than_mask = unique_expiries > tau
        far_expiry_idx = np.where(greater_than_mask)[0][0]

        near_skew = iv_surface[iv_surface.tau == unique_expiries[near_expiry_idx]][['strike', 'mark_iv']]
        far_skew = iv_surface[iv_surface.tau == unique_expiries[far_expiry_idx]][['strike', 'mark_iv']]

        iv = time_interpolation(near_skew, far_skew, tau, strike)

    return iv


def get_strike_from_delta(spot: float, tau: float, iv_surface: pd.DataFrame, rate: float,
                          is_call: bool, delta: float):
    """
    Calculates the strike price of the option for a given delta.
    :param spot: (float) Spot price.
    :param tau: (float) Time to maturity.
    :param iv_surface: (pd.DataFrame) Implied volatility surface from OptionsDefibuliser.
    :param rate: (float) Risk-free-rate.
    :param is_call: (bool) Whether the delta is calculated for call or for put option.
    :param delta: (float) Absolute value of elta of option required.
    :return: (list) Strike price and proximity to a true answer.
    """

    func = lambda strike: abs(black_scholes_delta_from_surface(spot=spot, strike=strike, tau=tau,
                                                               iv_surface=iv_surface, rate=rate,
                                                               is_call=is_call)) - abs(delta)

    root = fsolve(func, [spot])[0]

    output = [root, func(root)]

    return output
