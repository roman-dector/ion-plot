from pprint import pprint
from plot.graph import plot_graph
from plot.utils import (
    cast_data_to_dataframe,
    get_month_days_count,
    make_linear_regression,
    split_df_to_sun_moon,
    north_summer,
    north_winter,
)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from dal import select_coords_by_ursi
from dal.models import (
    select_2h_avr_for_day_with_sat_tec,
    get_f0f2_k_spread_for_month,
)


def calc_f0f2_k_mean_for_day(
    ursi: str,
    date: str,
):
    data = select_2h_avr_for_day_with_sat_tec(ursi, date)
    df = cast_data_to_dataframe(
        data,
        columns=['hour', 'f0f2', 'ion_tec', 'sat_tec', 'b0'],
        sat_tec=True,
    )

    sun, moon = split_df_to_sun_moon(df, ursi, date)

    mlr = lambda df, tec_name: make_linear_regression(
        y=[v**2 for v in df['f0f2']],
        x=df[tec_name],
        )
    
    ion_reg_sun = mlr(sun, 'ion_tec')
    ion_reg_moon = mlr(moon, 'ion_tec')
    ion_sun_k, ion_sun_k_err = ion_reg_sun.params[0], ion_reg_sun.bse[0]
    ion_moon_k, ion_moon_k_err = ion_reg_moon.params[0], ion_reg_moon.bse[0]

    sat_reg_sun = mlr(sun, 'sat_tec')
    sat_reg_moon = mlr(moon, 'sat_tec')
    sat_sun_k, sat_sun_k_err = sat_reg_sun.params[0], sat_reg_sun.bse[0]
    sat_moon_k, sat_moon_k_err = sat_reg_moon.params[0], sat_reg_moon.bse[0]

    return {
        'ion': {
            'sun': {'k': ion_sun_k, 'err': ion_sun_k_err},
            'moon': {'k': ion_moon_k, 'err': ion_moon_k_err},
        },
        'sat': {
            'sun': {'k': sat_sun_k, 'err': sat_sun_k_err},
            'moon': {'k': sat_moon_k, 'err': sat_moon_k_err},
        },
    }


# def get_f0f2_k_spread_for_month(
#     ursi: str,
#     month: int,
#     year: int,
# ):
#     ion_sun = []
#     ion_moon = []
#     sat_sun = []
#     sat_moon = []

#     ion_sun_err = []
#     ion_moon_err = []
#     sat_sun_err = []
#     sat_moon_err = []

#     for day in range(1, get_month_days_count(month, year) + 1):
#         str_month = f'0{month}' if month < 10 else f'{month}'
#         str_day = f'0{day}' if day < 10 else f'{day}'
#         date = f"{year}-{str_month}-{str_day}"

#         try:
#             k_mean = calc_f0f2_k_mean_for_day(ursi, date)

#             ion_sun.append(k_mean['ion']['sun']['k'])
#             ion_moon.append(k_mean['ion']['moon']['k'])
#             sat_sun.append(k_mean['sat']['sun']['k'])
#             sat_moon.append(k_mean['sat']['moon']['k'])

#             ion_sun_err.append(k_mean['ion']['sun']['err'])
#             ion_moon_err.append(k_mean['ion']['moon']['err'])
#             sat_sun_err.append(k_mean['sat']['sun']['err'])
#             sat_moon_err.append(k_mean['sat']['moon']['err'])
#         except:
#             continue

#     return {
#         'ion': {
#             'sun': {'k': ion_sun, 'err': ion_sun_err},
#             'moon': {'k':  ion_moon, 'err': ion_moon_err},
#         },
#         'sat': {
#             'sun': {'k': sat_sun, 'err': sat_sun_err},
#             'moon': {'k': sat_moon, 'err': sat_moon_err},
#         },
#     }


def get_f0f2_k_spread_for_summer_winter(
    ursi: str,
    year: int,
) -> list[float]:
    sum_ion_sun = []
    sum_ion_moon = []
    sum_sat_sun = []
    sum_sat_moon = []
    sum_ion_sun_err = []
    sum_ion_moon_err = []
    sum_sat_sun_err = []
    sum_sat_moon_err = []

    win_ion_sun = []
    win_ion_moon = []
    win_sat_sun = []
    win_sat_moon = []
    win_ion_sun_err = []
    win_ion_moon_err = []
    win_sat_sun_err = []
    win_sat_moon_err = []

    if select_coords_by_ursi(ursi)['lat'] > 0:
        sum_month = north_summer
        win_month = north_winter
    else:
        sum_month = north_winter
        win_month = north_summer

    for m in sum_month:
        try:
            k_mean = calc_f0f2_k_mean_for_month(ursi, m, year)

            sum_ion_sun.append(k_mean['ion']['sun']['k'])
            sum_ion_moon.append(k_mean['ion']['moon']['k'])
            sum_sat_sun.append(k_mean['sat']['sun']['k'])
            sum_sat_moon.append(k_mean['sat']['moon']['k'])
            sum_ion_sun_err.append(k_mean['ion']['sun']['err'])
            sum_ion_moon_err.append(k_mean['ion']['moon']['err'])
            sum_sat_sun_err.append(k_mean['sat']['sun']['err'])
            sum_sat_moon_err.append(k_mean['sat']['moon']['err'])
        except:
            continue

    for m in win_month:
        try:
            k_mean = calc_f0f2_k_mean_for_month(ursi, m, year)

            win_ion_sun.append(k_mean['ion']['sun']['k'])
            win_ion_moon.append(k_mean['ion']['moon']['k'])
            win_sat_sun.append(k_mean['sat']['sun']['k'])
            win_sat_moon.append(k_mean['sat']['moon']['k'])
            win_ion_sun_err.append(k_mean['ion']['sun']['err'])
            win_ion_moon_err.append(k_mean['ion']['moon']['err'])
            win_sat_sun_err.append(k_mean['sat']['sun']['err'])
            win_sat_moon_err.append(k_mean['sat']['moon']['err'])
        except:
            continue

    return {
        'ion': {
            'sum': {
                'sun': {'k': sum_ion_sun, 'err': sum_ion_sun_err},
                'moon': {'k':  sum_ion_moon, 'err': sum_ion_moon_err},
            },
            'win': {
                'sun': {'k': win_ion_sun, 'err': win_ion_sun_err},
                'moon': {'k':  win_ion_moon, 'err': win_ion_moon_err},
            },
        },
        'sat': {
            'sum': {
                'sun': {'k': sum_sat_sun, 'err': sum_sat_sun_err},
                'moon': {'k':  sum_sat_moon, 'err': sum_sat_moon_err},
            },
            'win': {
                'sun': {'k': win_sat_sun, 'err': win_sat_sun_err},
                'moon': {'k':  win_sat_moon, 'err': win_sat_moon_err},
            },
        },
    }


def get_f0f2_k_spread_for_year(ursi: str, year: int):
    ion_sun = []
    ion_moon = []
    ion_sun_err = []
    ion_moon_err = []

    sat_sun = []
    sat_moon = []
    sat_sun_err = []
    sat_moon_err = []

    for m in range(1, 13):
        try:
            k_mean = calc_f0f2_k_spread_for_month(ursi, m, year)

            ion_sun.append(k_mean['ion']['sun']['k'])
            ion_moon.append(k_mean['ion']['moon']['k'])
            sat_sun.append(k_mean['sat']['sun']['k'])
            sat_moon.append(k_mean['sat']['moon']['k'])
            ion_sun_err.append(k_mean['ion']['sun']['err'])
            ion_moon_err.append(k_mean['ion']['moon']['err'])
            sat_sun_err.append(k_mean['sat']['sun']['err'])
            sat_moon_err.append(k_mean['sat']['moon']['err'])
        except:
            continue

    return {
        'ion': {
            'sun': {'k': ion_sun, 'err': ion_sun_err},
            'moon': {'k':  ion_moon, 'err': ion_moon_err},
        },
        'sat': {
            'sun': {'k': sat_sun, 'err': sat_sun_err},
            'moon': {'k': sat_moon, 'err': sat_moon_err},
        },
    }



# def calc_f0f2_k_mean_for_month(
#     ursi: str,
#     month: int,
#     year: int,
# ):
#     k_mean = calc_f0f2_k_spread_for_month(ursi, month, year)

#     ion_sun_len = len(k_mean['ion']['sun']['k'])
#     ion_moon_len = len(k_mean['ion']['moon']['k'])
#     sat_sun_len = len(k_mean['sat']['sun']['k'])
#     sat_moon_len = len(k_mean['sat']['moon']['k'])

#     return {
#         'ion': {
#             'sun': {
#                 'k': sum(k_mean['ion']['sun']['k']) / ion_sun_len,
#                 'err': sum(k_mean['ion']['sun']['err']) / ion_sun_len,
#             },
#             'moon': {
#                 'k':  sum(k_mean['ion']['moon']['k']) / ion_moon_len,
#                 'err': sum(k_mean['ion']['moon']['err']) / ion_moon_len,
#             },
#         },
#         'sat': {
#             'sun': {
#                 'k': sum(k_mean['sat']['sun']['k']) / sat_sun_len,
#                 'err': sum(k_mean['sat']['sun']['err']) / sat_sun_len,
#             },
#             'moon': {
#                 'k': sum(k_mean['sat']['moon']['k']) / sat_moon_len,
#                 'err': sum(k_mean['sat']['moon']['err']) / sat_moon_len,
#             },
#         },
#     }


def plot_f0f2_k_spread_for_month(
    ursi: str,
    month: int,
    year: int,
):
    coords = select_coords_by_ursi(ursi)
    ion_sun_k, ion_moon_k, sat_sun_k, sat_moon_k = get_f0f2_k_spread_for_month(ursi, month, year)
    
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 6))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year} Month: {month}",
        fontsize=18, y=0.98,
    )

    ax[0][0].set_title('Ion-Sun', fontsize=15)
    ax[0][1].set_title('Ion-Moon', fontsize=15)
    ax[1][0].set_title('Sat-Sun', fontsize=15)
    ax[1][1].set_title('Sat-Moon', fontsize=15)
    
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    
    ax[0][0].set_xlim(None, 10)
    ax[0][0].set_ylim(None, 30)
    ax[0][1].set_xlim(None, 10)
    ax[0][1].set_ylim(None, 30)
    ax[1][0].set_xlim(None, 10)
    ax[1][0].set_ylim(None, 30)
    ax[1][1].set_xlim(None, 10)
    ax[1][1].set_ylim(None, 30)

    sns.histplot(ion_sun_k, kde=True, ax=ax[0][0])
    sns.histplot(ion_moon_k, kde=True, ax=ax[0][1])
    sns.histplot(sat_sun_k, kde=True, ax=ax[1][0])
    sns.histplot(sat_moon_k, kde=True, ax=ax[1][1])
    
    mu_sum_sun, std_sum_sun = norm.fit(ion_sun_k)
    textstr_ion_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_sum_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_sun, )))
    
    mu_sum_moon, std_sum_moon = norm.fit(ion_moon_k)
    textstr_ion_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_sum_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_moon, )))
    
    mu_win_sun, std_win_sun = norm.fit(sat_sun_k)
    textstr_sat_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_win_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_sun, )))
    
    mu_win_moon, std_win_moon = norm.fit(sat_moon_k)
    textstr_sat_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_win_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, textstr_ion_sun, transform=ax[0][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, textstr_ion_moon, transform=ax[0][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, textstr_sat_sun, transform=ax[1][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, textstr_sat_moon, transform=ax[1][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_f0f2_k_spread_for_summer_winter(
    ursi: str,
    year: int=2019,
    sat_tec: bool=False
):
    coords = select_coords_by_ursi(ursi)

    sum_result, win_result = count_f0f2_k_spreading_for_summer_winter(ursi, year)
    sum_sun_result, sum_moon_result = sum_result
    win_sun_result, win_moon_result = win_result

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.98,
    )
    
    ax[0][0].set_title('Summer - Sun', fontsize=15)
    ax[0][1].set_title('Summer - Moon', fontsize=15)
    ax[1][0].set_title('Winter - Sun', fontsize=15)
    ax[1][1].set_title('Winter - Moon', fontsize=15)
    
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    
    ax[0][0].set_xlim(None, 10)
    ax[0][0].set_ylim(None, 30)
    ax[0][1].set_xlim(None, 10)
    ax[0][1].set_ylim(None, 30)
    ax[1][0].set_xlim(None, 10)
    ax[1][0].set_ylim(None, 30)
    ax[1][1].set_xlim(None, 10)
    ax[1][1].set_ylim(None, 30)

    sns.histplot(sum_sun_result, kde=True, ax=ax[0][0])
    sns.histplot(sum_moon_result, kde=True, ax=ax[0][1])
    sns.histplot(win_sun_result, kde=True, ax=ax[1][0])
    sns.histplot(win_moon_result, kde=True, ax=ax[1][1])
    
    
    mu_sum_sun, std_sum_sun = norm.fit(sum_sun_result)
    textstr_sum_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_sum_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_sun, )))
    
    mu_sum_moon, std_sum_moon = norm.fit(sum_moon_result)
    textstr_sum_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_sum_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_moon, )))
    
    mu_win_sun, std_win_sun = norm.fit(win_sun_result)
    textstr_win_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_win_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_sun, )))
    
    mu_win_moon, std_win_moon = norm.fit(win_moon_result)
    textstr_win_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_win_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_moon, )))


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, textstr_sum_sun, transform=ax[0][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, textstr_sum_moon, transform=ax[0][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, textstr_win_sun, transform=ax[1][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, textstr_win_moon, transform=ax[1][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_f0f2_k_spreading_for_year(ursi: str, year: int=2019):
    coords = select_coords_by_ursi(ursi)
    
    sun_range, moon_range = count_f0f2_k_spreading_for_year(ursi, year)
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.98,
    )
    
    
    ax[0].set_title('Sun', fontsize=15)
    ax[1].set_title('Moon', fontsize=15)
    ax[0].grid()
    ax[1].grid()
    
    ax[0].set_xlim(None, 10)
    ax[0].set_ylim(None, 50)
    ax[1].set_xlim(None, 10)
    ax[1].set_ylim(None, 50)

    sns.histplot(sun_range, kde=True, ax=ax[0])
    sns.histplot(moon_range, kde=True, ax=ax[1])
    
    mu_sun, std_sun = norm.fit(sun_range)
    textstr_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_sun, ),
    r'$\sigma^2=%.2f$' % (std_sun, )))
    
    mu_moon, std_moon = norm.fit(moon_range)
    textstr_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_moon, ),
    r'$\sigma^2=%.2f$' % (std_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0].text(0.05, 0.95, textstr_sun, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, textstr_moon, transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_k_spreading_lat_split_month_graph(month: int, year: int, stations_list: list[str]):
    k_sun_range = []
    k_moon_range = []
    lat_range = []

    for s in stations_list:
        try:
            k = calc_f0f2_k_mean_for_month(s, month, year)

            k_sun_range.append(sum(k[0])/len(k[0]))
            k_moon_range.append(sum(k[1])/len(k[1]))
            lat_range.append(select_coords_by_ursi(s)['lat'])
        except Exception as ex:
            print(ex)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,6))
    
    fig.suptitle(f"Year: {year}, Month: {month}", fontsize=20, y=0.96)

    ax[0].grid()
    ax[1].grid()
    
    ax[0].set_xlim(None, 60)
    ax[0].set_ylim(None, 10)
    ax[1].set_xlim(None, 60)
    ax[1].set_ylim(None, 10)

    plot_graph(
        ax[0], lat_range, k_sun_range,
        'lat', 'k', 'Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[1], lat_range, k_moon_range,
        'lat', 'k', 'Moon', color='purple',
        edgecolor='b', const=True,
    )


def plot_k_spreading_lat_sum_win_split_graph(month: int, year: int, stations_list: list[str]):
    k_sum_sun_range = []
    k_sum_moon_range = []
    k_win_sun_range = []
    k_win_moon_range = []
    lat_range = []

    for s in stations_list:
        try:
            k = calc_f0f2_k_mean_for_summer_winter(s, year)

            k_sum_sun_range.append(k[0][0])
            k_sum_moon_range.append(k[0][1])
            k_win_sun_range.append(k[1][0])
            k_win_moon_range.append(k[1][1])

            lat_range.append(select_coords_by_ursi(s)['lat'])
        except Exception as ex:
            print(ex)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15,6))
    
    fig.suptitle(f"Year: {year}, Month: {month}", fontsize=20, y=0.96)

    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()

    ax[0][0].set_xlim(None, 60)
    ax[0][1].set_ylim(None, 10)
    ax[1][0].set_xlim(None, 60)
    ax[1][1].set_ylim(None, 10)

    plot_graph(
        ax[0][0], lat_range, k_sum_sun_range,
        'lat', 'k', 'Sum-Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[0][1], lat_range, k_sum_moon_range,
        'lat', 'k', 'Win-Moon', color='purple',
        edgecolor='b', const=True,
    )
    plot_graph(
        ax[1][0], lat_range, k_win_sun_range,
        'lat', 'k', 'Sum-Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[1][1], lat_range, k_win_moon_range,
        'lat', 'k', 'Win-Moon', color='purple',
        edgecolor='b', const=True,
    )
