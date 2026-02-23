import pandas as pd

def simple_moving_average(data_frame, ohlc, time_frame):
    """
    Calculate Simple Moving Average (SMA) for a given OHLC column and time frame.

    Parameters:
        data_frame (pd.DataFrame): Your OHLC DataFrame
        ohlc (str): Column name (e.g., 'Close', 'Open')
        time_frame (int): Number of periods for SMA

    Returns:
        pd.DataFrame: Original DataFrame with new SMA column added
    """
    sma_raw = data_frame[ohlc].rolling(window=time_frame).mean()
    data_frame[f"sma_{ohlc.lower()}_{time_frame}"] = (sma_raw - data_frame[ohlc]) / data_frame[ohlc]
    return data_frame

def relative_strength_index(df, ohlc, time_frame):
    """
    Calculate Relative Strength Index (RSI) for a given OHLC column and time frame manually.

    Parameters:
        df (pd.DataFrame): OHLC DataFrame
        ohlc (str): Column name, e.g., 'Close'
        time_frame (int): Number of periods for RSI (commonly 14)

    Returns:
        pd.DataFrame: DataFrame with a new RSI column
    """
    delta = df[ohlc].diff()  # daily change
    gain = delta.clip(lower=0)   # positive changes
    loss = -delta.clip(upper=0)  # negative changes as positive

    # Calculate average gain/loss using exponential moving average
    avg_gain = gain.rolling(window=time_frame, min_periods=time_frame).mean()
    avg_loss = loss.rolling(window=time_frame, min_periods=time_frame).mean()

    # Use Wilder's smoothing after first window
    avg_gain = avg_gain.combine_first(gain.ewm(alpha=1/time_frame, min_periods=time_frame, adjust=False).mean())
    avg_loss = avg_loss.combine_first(loss.ewm(alpha=1/time_frame, min_periods=time_frame, adjust=False).mean())

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[f"rsi_{ohlc.lower()}_{time_frame}"] = rsi
    return df

def momentum(df, ohlc, time_frame):
    """
    Calculate Momentum (MOM) for a given OHLC column and time frame manually.

    Parameters:
        df (pd.DataFrame): OHLC DataFrame
        ohlc (str): Column name, e.g., 'Close'
        time_frame (int): Number of periods for MOM

    Returns:
        pd.DataFrame: DataFrame with a new MOM column
    """
    df[f"mom_{ohlc.lower()}_{time_frame}"] = df[ohlc].pct_change(periods=time_frame) * 100
    return df

def bollinger_bands(df, ohlc, time_frame, std_factor=2):
    """
    Calculate Bollinger Bands in % relative to the middle band (SMA).
    Adds 3 columns: Lower_pct, Middle_pct (0), Upper_pct

    Parameters:
        df (pd.DataFrame): OHLC DataFrame
        ohlc (str): Column name, e.g., 'Close'
        time_frame (int): Number of periods for SMA (middle band)
        std_factor (float): Number of standard deviations for upper/lower bands

    Returns:
        pd.DataFrame: DataFrame with new BBANDS % columns
    """
    # Middle band = SMA
    mb = df[ohlc].rolling(window=time_frame).mean()
    # Standard deviation
    std = df[ohlc].rolling(window=time_frame).std()

    # Percent Bollinger Bands relative to middle band
    df[f"bbands_{ohlc.lower()}_{time_frame}_m_pct"] = 0  # middle band as 0%
    df[f"bbands_{ohlc.lower()}_{time_frame}_u_pct"] = ((mb + std_factor * std) - mb) / mb * 100
    df[f"bbands_{ohlc.lower()}_{time_frame}_l_pct"] = ((mb - std_factor * std) - mb) / mb * 100
    df[f"bbands_{ohlc.lower()}_{time_frame}_width_pct"] = df[f"bbands_{ohlc.lower()}_{time_frame}_u_pct"] - df[f"bbands_{ohlc.lower()}_{time_frame}_l_pct"]

    return df

def exponential_moving_average(series, span):
    """
    Calculate EMA manually using pandas ewm.
    """
    return series.ewm(span=span, adjust=False).mean()

def macd(df, ohlc, fast=12, slow=26, signal=9):
    """
    Calculate MACD in % relative to the slow EMA for a given OHLC column.

    Adds 3 columns:
        - MACD_pct line
        - Signal_pct line
        - Histogram_pct

    Parameters:
        df (pd.DataFrame): OHLC DataFrame
        ohlc (str): Column name, e.g., 'Close'
        fast (int): Fast EMA period (default 12)
        slow (int): Slow EMA period (default 26)
        signal (int): Signal EMA period (default 9)

    Returns:
        pd.DataFrame: DataFrame with MACD % columns
    """
    # Calculate EMAs
    ema_fast = exponential_moving_average(df[ohlc], fast)
    ema_slow = exponential_moving_average(df[ohlc], slow)

    # MACD line
    macd_line = ema_fast - ema_slow
    # Signal line
    signal_line = exponential_moving_average(macd_line, signal)
    # Histogram
    histogram = macd_line - signal_line

    # Convert to % relative to slow EMA
    df[f"macd_{ohlc.lower()}_pct"] = (macd_line / ema_slow) * 100
    df[f"macd_signal_{ohlc.lower()}_pct"] = (signal_line / ema_slow) * 100
    df[f"macd_hist_{ohlc.lower()}_pct"] = (histogram / ema_slow) * 100

    return df

def volatility(df, ret_col='ret_1', windows=[5, 10, 20]):
    """
    Calculate rolling volatility (% std deviation) over given windows.

    Parameters:
        df (pd.DataFrame): DataFrame with returns column
        ret_col (str): Column name for returns, in %
        windows (list): List of rolling window sizes

    Returns:
        pd.DataFrame: DataFrame with new volatility columns
    """
    for w in windows:
        df[f"volatility_{w}d_pct"] = df[ret_col].rolling(window=w).std()
    return df

def add_technical_indicators(df, ohlc="Close", time_frame=10):
    """
    Add common technical indicators: SMA20, RSI14, MOM14, BBANDS20
    """
    df = simple_moving_average(df, ohlc, time_frame)
    df = relative_strength_index(df, ohlc, time_frame)
    df = momentum(df, ohlc, time_frame)
    df = bollinger_bands(df, ohlc, time_frame)
    df = macd(df, ohlc)
    return df
