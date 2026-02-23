import pandas as pd
import yfinance as yf
from datetime import date
from indicators import add_technical_indicators, volatility
from returns import get_returns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from helpers.utility_helper import get_previous_date
import joblib

from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

df = yf.download("^NSEI", start="1970-01-01", end=date.today())

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


df= add_technical_indicators(df, ohlc="Close", time_frame=10)
df = get_returns(df, 'Close', 1)
df = get_returns(df, 'Close', 5)
df = get_returns(df, 'Close', 10)


df = volatility(df)

df['next_close'] = df['Close'].shift(-1)

clone_df = df.copy()
df.dropna(inplace=True)

print(df.columns)
print(df.head(50))
print(df.tail(50))

features = [
    'Close', 'High', 'Low', 'Open', 'Volume', 'sma_close_10',
    'rsi_close_10', 'mom_close_10', 'bbands_close_10_m_pct',
    'bbands_close_10_u_pct', 'bbands_close_10_l_pct',
    'bbands_close_10_width_pct', 'macd_close_pct', 'macd_signal_close_pct',
    'macd_hist_close_pct', 'ret_1', 'ret_5', 'ret_10', 'volatility_5d_pct',
    'volatility_10d_pct', 'volatility_20d_pct'
    ]

X = df[features]
y = df['next_close']


split_date = get_previous_date(3)

X_train = X[X.index < split_date]
X_test  = X[X.index >= split_date]

y_train = y[y.index < split_date]
y_test  = y[y.index >= split_date]

# param_grid = {
#     "n_estimators": [100, 200, 300],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "max_depth": [3, 5, 7],
#     "subsample": [0.8, 1.0],
# }

# grid = GridSearchCV(
#     xgb.XGBRegressor(random_state=42),
#     param_grid,
#     scoring="neg_mean_squared_error",
#     cv=5,
#     verbose=1
# )

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.5,
    max_depth=2,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
], memory=None)

print('🚀 Model Training begins... 🔂 ')
pipeline.fit(X_train, y_train)
print('🚀 Model is trained 🙂  ')

joblib.dump(pipeline, "../models/NiFTY_next_open_prediction.pkl")