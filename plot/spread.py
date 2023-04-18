from plot.utils import (
    cast_data_to_dataframe,
    get_month_days_count,
    make_linear_regression,
    split_df_to_sun_moon,
    north_summer,
    north_winter,
)
from scipy.stats import norm
from dal import select_hour_avr_for_day, select_coords_by_ursi


def count_f0f2_k_for_day(
    ursi: str,
    date: str,
) -> tuple[float]:
    df = cast_data_to_dataframe(
        select_hour_avr_for_day(ursi, date),
        columns=['hour', 'f0f2', 'tec', 'b0'],
    )
    
    sun, moon = split_df_to_sun_moon(df, ursi, date)
    
    mlr = lambda df: make_linear_regression([v**2 for v in df['f0f2']], df['tec']).params[0]


    return mlr(sun), mlr(moon)


def count_f0f2_k_spreading_for_month(
    ursi: str,
    month: int,
    year: int=2019,
) -> list[float]:
    sun_result: list[float] = []
    moon_result: list[float] = []
    
    for day in range(1, get_month_days_count(month) + 1):
        str_month = f'0{month}' if month < 10 else f'{month}'
        str_day = f'0{day}' if day < 10 else f'{day}'
        date = f"{year}-{str_month}-{str_day}"
        
        try:
            k = count_f0f2_k_for_day(ursi, date)

            sun_result.append(k[0])
            moon_result.append(k[1])
        except:
            continue

    return sun_result, moon_result


def count_f0f2_k_spreading_for_summer_winter(
    ursi: str,
    year: int=2019,
) -> list[float]:
    sum_sun_result: list[float] = []
    sum_moon_result: list[float] = []
        
    win_sun_result: list[float] = []
    win_moon_result: list[float] = []

    if select_coords_by_ursi(ursi)['lat'] > 0:
        sum_month = north_summer
        win_month = north_winter
    else:
        sum_month = north_winter
        win_month = north_summer
    
    for m in sum_month:
        try:
            sun, moon = count_f0f2_k_spreading_for_month(ursi, m, year)
            sum_sun_result = [*sun, *sum_sun_result]
            sum_moon_result = [*moon, *sum_moon_result]
        except:
            pass

    for m in win_month:
        try:
            sun, moon = count_f0f2_k_spreading_for_month(ursi, m, year)
            win_sun_result = [*sun, *win_sun_result]
            win_moon_result = [*moon, *win_moon_result]
        except:
            pass
        
    return (sum_sun_result, sum_moon_result), (win_sun_result, win_moon_result)


def count_f0f2_k_spreading_for_year(ursi: str, year: int=2019) -> list[float]:
    sun_result: list[float] = []
    moon_result: list[float] = []
        
    for m in range(1, 13):
        try:
            sun, moon = count_f0f2_k_spreading_for_month(ursi, m, year)
            sun_result = [*sun_result, *sun]
            moon_result = [*moon_result, *moon]
        except:
            pass
    
    return sun_result, moon_result

def calc_f0f2_k_mean_for_day(
    ursi: str,
    date: str,
) -> tuple[float]:
    df = cast_data_to_dataframe(
        select_hour_avr_for_day(ursi, date),
        columns=['hour', 'f0f2', 'tec', 'b0'],
    )
    
    sun, moon = split_df_to_sun_moon(df, ursi, date)
    
    mlr = lambda df: make_linear_regression([v**2 for v in df['f0f2']], df['tec'])
    
    reg_sun = mlr(sun)
    reg_moon = mlr(moon)
    sun_k, sun_k_err = reg_sun.params[0], reg_sun.bse[0]
    moon_k, moon_k_err = reg_moon.params[0], reg_moon.bse[0]

    return ((sun_k, sun_k_err), (moon_k, moon_k_err))


def calc_f0f2_k_mean_for_month(
    ursi: str,
    month: int,
    year: int=2019,
) -> tuple[float]:
    sun_range, moon_range = count_f0f2_k_spreading_for_month(ursi, month, year)
    
    sun_mean, sun_std_err = norm.fit(sun_range)
    moon_mean, moon_std_err = norm.fit(moon_range)

    return ((sun_mean, sun_std_err), (moon_mean, moon_std_err))


def calc_f0f2_k_mean_for_summer_winter(
    ursi: str,
    year: int=2019 ,
) -> tuple[tuple[float]]:
    sum_range, win_range = count_f0f2_k_spreading_for_summer_winter(ursi, year)
    
    sum_sun_mean, sum_sun_std_err = norm.fit(sum_range[0])
    sum_moon_mean, sum_moon_std_err = norm.fit(sum_range[1])
    win_sun_mean, win_sun_std_err = norm.fit(win_range[0])
    win_moon_mean, win_moon_std_err = norm.fit(win_range[1])
    
    return (
        ((sum_sun_mean, sum_sun_std_err), (sum_moon_mean, sum_moon_std_err)),
        ((win_sun_mean, win_sun_std_err), (win_moon_mean, win_moon_std_err)),
    )


def calc_f0f2_k_mean_for_year(
    ursi: str,
    year: int=2019 ,
) -> tuple[float]:
    sun_range, moon_range = count_f0f2_k_spreading_for_year(ursi, year)
    
    sun_mean, sun_std_err = norm.fit(sun_range)
    moon_mean, moon_std_err = norm.fit(moon_range)

    return ((sun_mean, sun_std_err), (moon_mean, moon_std_err))