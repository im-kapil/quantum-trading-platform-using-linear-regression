import pandas as pd
import yfinance as yf
from datetime import date
from indicators import add_technical_indicators, volatility
from returns import get_returns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from helpers.utility_helper import get_previous_date
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb
import numpy as np

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
    'Close', 'High', 'Low', 'Open', 'Volume', 
    'sma_close_10',
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


model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,   # smaller learning rate
    max_depth=3,           # shallow trees
    subsample=0.8,         # optional: adds regularization
    colsample_bytree=0.8,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
], memory=None)

# pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", LinearRegression())
# ])


print('🚀 Model Training begins... 🔂 ')
pipeline.fit(X_train, y_train)
print('🚀 Model is trained 🙂  ')

joblib.dump(pipeline, "../models/NiFTY_next_open_prediction.pkl")

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

direction_accuracy = np.mean(np.sign(y_pred - X_test['Close']) == np.sign(y_test - X_test['Close']))
print("Direction Accuracy:", direction_accuracy)