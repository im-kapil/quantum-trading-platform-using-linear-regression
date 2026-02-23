import pandas as pd
import yfinance as yf
from datetime import date
from src.indicators import add_technical_indicators, volatility
from src.returns import get_returns

df = yf.download("^NSEI", start="1970-01-01", end=date.today())

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


df= add_technical_indicators(df, ohlc="Close", time_frame=10)
df = get_returns(df, 'Close', 1)
df = get_returns(df, 'Close', 5)
df = get_returns(df, 'Close', 10)


df = volatility(df)

print(df.columns)
print(df.head(50))

