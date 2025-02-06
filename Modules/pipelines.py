from sklearn.preprocessing import FunctionTransformer
import pandas as pd

# Creating Functions to convert to datetime

def to_dt(series, fmt):

    """
    Converts a series to a dt object

    Args:
        series (pd.Series): Series to convert
        fmt (str): date format of series

    Returns:
        pd.Series: converted series
    """    
    
    return pd.to_datetime(series, format=fmt)

# Encapsulating function as transformer

hour_min_converter = FunctionTransformer(to_dt, kw_args={"fmt": "%H:%M:%S"})