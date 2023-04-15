from datetime import datetime
from calendar import monthrange

import pandas as pd
import statsmodels.api as sm

from suntime import Sun, SunTimeException
from statsmodels.regression.linear_model import (
    RegressionResultsWrapper as RRW
)

from dal.models import(
    IonData,
    transform_data,
    select_middle_lat_stations,
)


def get_sunrise_sunset(date, coords):
    sun = Sun(coords['lat'], coords['long'])
    abd = datetime.strptime(date, '%Y-%m-%d').date()
    
    try:
        sunrise = sun.get_sunrise_time(abd).time().strftime('%H')
        sunset = sun.get_sunset_time(abd).time().strftime('%H')

        return sunrise, sunset
    except SunTimeException as e:
        print(f"Error: {e}")


def cast_data_to_dataframe(data: IonData, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(transform_data(data), columns=columns)


def convert_iso_to_day_of_year(date: str) -> int:
    return datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday


def get_month_days_count(month: int, year: int=2019) -> int:
    return monthrange(year, month)[1]


def make_linear_regression(y: list[float], x: list[float]) -> RRW:
    sm.add_constant(x)
    return sm.OLS(y, x).fit()

