"""
Temp module docstring
"""
import pandas as pd
from scipy.interpolate import interp1d

# pylint: disable=invalid-name
def time_interpolation(data, x_new):  # pragma: no cover
    """
    Interpolates on datetimes using input dataframe to find outputs for new time points

    :param data (Pandas Series): Series indexed by datetime64['ns',utc] with 1 column of data
    :param x_new (Pandas Series): Series containing datetime64['ns',utc] points for interpolation

    :return (Pandas Dataframe): DataFrame containing interpolated points
    """
    x = data.index.values.astype(int) / 1e9
    y = data.values

    new_times = x_new.values.astype(int) / 1e9
    f = interp1d(x, y)
    out = f(new_times)
    return pd.DataFrame(out, columns=['interp_data'])
