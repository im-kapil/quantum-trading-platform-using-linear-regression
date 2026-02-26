from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_previous_date(date_to_year):
    today = datetime.now()
    date = today - relativedelta(years=date_to_year)
    return date.strftime("%Y-%m-%d")

import numpy as np
import pandas as pd

def drop_highly_correlated_features(df, threshold=0.9):
    # Compute correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return df.drop(columns=to_drop)