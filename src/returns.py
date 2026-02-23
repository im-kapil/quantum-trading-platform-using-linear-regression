def get_returns(data_frame, ohlc, time_frame):
    """
    Calculate the percent return for a given OHLC column over a specified time frame.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame containing price data.
        ohlc (str): The column name to calculate returns on (e.g., 'Close').
        time_frame (int): Number of periods over which to calculate the return.

    Returns:
        pd.DataFrame: Original DataFrame with a new column added in the format 'ret_<time_frame>', 
                      containing the percent return over the specified period.

    Example:
        df = get_returns(df, 'Close', 5)
        # Adds a column 'ret_5' with 5-period percent returns.
    """
    data_frame[f'ret_{time_frame}'] = data_frame[ohlc].pct_change(periods=time_frame) * 100
    return data_frame