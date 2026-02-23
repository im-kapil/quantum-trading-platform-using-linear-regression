import pandas as pd
import yfinance as yf
from datetime import date
from src.indicators import add_technical_indicators, volatility
from src.returns import get_returns
import joblib

model = joblib.load("./models/NiFTY_next_open_prediction.pkl")

df = yf.download("^NSEI", start="1970-01-01", end=date.today())

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


df= add_technical_indicators(df, ohlc="Close", time_frame=10)
df = get_returns(df, 'Close', 1)
df = get_returns(df, 'Close', 5)
df = get_returns(df, 'Close', 10)


df = volatility(df)

last_row = df.iloc[[-1]]

features = last_row[[
    'Close', 'High', 'Low', 'Open', 'Volume', 
    'sma_close_10',
    'rsi_close_10', 'mom_close_10', 'bbands_close_10_m_pct',
    'bbands_close_10_u_pct', 'bbands_close_10_l_pct',
    'bbands_close_10_width_pct', 'macd_close_pct', 'macd_signal_close_pct',
    'macd_hist_close_pct', 'ret_1', 'ret_5', 'ret_10', 'volatility_5d_pct',
    'volatility_10d_pct', 'volatility_20d_pct'
    ]]

print(features)

prediction = model.predict(features)
print(prediction)



