from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_previous_date(date_to_year):
    today = datetime.now()
    date = today - relativedelta(years=date_to_year)
    return date.strftime("%Y-%m-%d")