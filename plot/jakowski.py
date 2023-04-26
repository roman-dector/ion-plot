from math import sin, cos, pi, sqrt, radians
from datetime import datetime as dt

from aacgmv2 import get_aacgm_coord
from dateutil import tz
from timezonefinder import TimezoneFinder

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from plot.utils import (
    convert_iso_to_day_of_year,
    get_sunrise_sunset,
)
from plot.graph import ( plot_graph )
from dal.models import (
    select_coords_by_ursi,
    select_solar_flux_day_mean,
    select_solar_flux_81_mean,
    select_f0f2_sat_tec,
    select_ion_tec_sat_tec,
)

c1 = 0.25031
c2 = -0.10451
c3 = 0.12037
c4 = -0.01268
c5 = -0.00779
c6 = 0.03171
c7 = -0.11763
c8 = 0.06199
c9 = -0.01147
c10 = 0.03417
c11 = 302.44989
c12 = 0.00474


def utc_to_local(date, time, lat, long):
    tf = TimezoneFinder()

    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(tf.timezone_at(lng=long, lat=lat))

    utc = dt.strptime(' '.join([date, time]), '%Y-%m-%d %H:%M:%S')

    utc = utc.replace(tzinfo=from_zone)
    local_dt = utc.astimezone(to_zone)
    
    h = float(dt.strftime(local_dt, '%H'))

    return h + (int(dt.strftime(local_dt, '%M')) / 60)


# lat in radians
def calc_cos_hi(lat, delta):
    return sin(lat) * sin(delta) + cos(lat) * cos(delta)


def calc_cos_hi2(cos_hi, lat, delta):
    return cos_hi - (2 * lat / pi) * sin(delta)


def calc_cos_hi3(cos_hi, SD=0.8):
    return sqrt(cos_hi + SD)

# lat in degree
def F1(
    date: str,
    time: str,
    lat,
    long,
    LTd=2,
    LTsd=9,
    LTtd: int=4,
    LTqd: int=6,
):
    LT = utc_to_local(date, time, lat, long)
    
    lat = radians(lat)
    DVm = 2*pi*(LT - LTd)/24
    SDVm = 2*pi*(LT - LTsd)/12
    TDVm = 2*pi*(LT - LTtd)/8
    QDVm = 2*pi*(LT - LTqd)/6
    
    
    N = convert_iso_to_day_of_year(date)
    delta = radians(-23.44 * cos(radians(360/365 * (N + 10))))
    
    cos_hi = calc_cos_hi(lat, delta)
    cos_hi2 = calc_cos_hi2(cos_hi, lat, delta)
    cos_hi3 = calc_cos_hi3(cos_hi)
    
    return cos_hi3 + (
        c1 * cos(DVm) +
        c2 * cos(SDVm) +
        c3 * cos(TDVm) +
        c4 * cos(QDVm)
    ) * cos_hi2


def F2(date, doya=340, doysa=360):
    doy = convert_iso_to_day_of_year(date)
    AVm = 2 * pi * (doy - doya) / 365.25
    SAVm = 4 * pi * (doy - doysa) / 365.25
    
    return 1 + c5 * cos(AVm) + c6 * cos(SAVm)

# lat in degree
def F3(lat: float, long: float, date: str):
    geom_lat = radians(
        get_aacgm_coord(lat, long, 0, dt.strptime(date, '%Y-%m-%d'))[0]
    )
    return (
        1 + c7*cos(2*pi*geom_lat/180) +
        c8*sin(2*pi*geom_lat/80) +
        c9*sin(2*pi*geom_lat/55) +
        c10*cos(2*pi*geom_lat/40)
    )


def calc_F10_1(F10, F10_81):
    return (0.8*F10 + 1.2*F10_81)/2


def calc_F10_2(F10_1):
    return (F10_1 - 90)**2


def F4(date: str):
    F10 = select_solar_flux_day_mean(date)
    F10_81 = select_solar_flux_81_mean(date)
    
    return c11 + c12*(calc_F10_2(calc_F10_1(F10, F10_81)))


def calc_tau(
    date: str,
    time: str,
    lat: float,
    long: float,
):
    if long > 180.0:
        long = long - 360
    return (
        F1(date, time, lat, long) *
        F2(date) *
        F3(lat, long, date) *
        F4(date)
    )


Z = 1/12.4

def calc_k(
    date: str,
    time: str,
    lat: float,
    long: float,
):
    return 1e4 * Z / calc_tau(date, time, lat, long)


def calc_f0F2(k, TEC):
    return sqrt(k * TEC)


def split_to_sun_moon(data, ursi, date):
    sunrise, sunset = get_sunrise_sunset(date, select_coords_by_ursi(ursi))

    if sunrise < sunset:
        # sun = df[(hour >= sunrise) & (hour < sunset)]
        # moon = df[(hour < sunrise) | (hour >= sunset)]
        sun = [r for r in data if (r[0] >= sunrise) and (r[0] < sunset)]
        moon = [r for r in data if (r[0] < sunrise) or (r[0] >= sunset)]
    else:
        # sun = df[(hour >= sunrise) | (hour < sunset)]
        # moon = df[(hour < sunrise) & (hour >= sunset)]
        sun = [r for r in data if (r[0] >= sunrise) or (r[0] < sunset)]
        moon = [r for r in data if (r[0] < sunrise) and (r[0] >= sunset)]
        
    return sun, moon



def plot_compare_jmodel_ion_f0f2(ursi, date):
    coords = select_coords_by_ursi(ursi)
    lat, long = coords['lat'], coords['long']

    row_data = select_f0f2_sat_tec(ursi, date)
    hour = [r[0] for r in row_data]
    f0f2 = [r[1] for r in row_data]
    sat_tec = [r[2] for r in row_data]

    jmodel_f0f2 = [
        calc_f0F2(calc_k(date, r[0]+':00:00', lat, long), r[2])
        for r in row_data
    ]

    k = [round(r[1]**2/r[2], 1) for r in row_data]
    jmodel_k = [round(calc_k(date, h+':00:00', lat, long), 1) for h in hour]

    r1 = round(pearsonr(f0f2, jmodel_f0f2)[0], 2)
    r2 = round(pearsonr(k, jmodel_k)[0], 2)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,8))

    ax[0].set_title(
        f"{ursi}, lat: {coords['lat']}, long: {coords['lat']}, r={r1}",
        fontsize=15,
    )
    ax[1].set_title(
        f"{ursi}, lat: {coords['lat']}, long: {coords['lat']}, r={r2}",
        fontsize=15,
    )

    plot_graph(
        ax=ax[0],
        x_ax=hour,
        y_ax=f0f2,
        x_label='hour',
        y_label='$f_0F_2$',
        title='green: real',
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[0],
        x_ax=hour,
        y_ax=jmodel_f0f2,
        x_label='hour',
        y_label='$f_0F_2$',
        title='blue: model',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )

    plot_graph(
        ax=ax[1],
        x_ax=hour,
        y_ax=k,
        x_label='hour',
        y_label='k',
        title='green: real',
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[1],
        x_ax=hour,
        y_ax=jmodel_k,
        x_label='hour',
        y_label='k',
        title='blue: model',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )


def plot_compare_ion_tec_sat_tec(ursi, date, split: bool=False):
    coords = select_coords_by_ursi(ursi)
    row_data = select_ion_tec_sat_tec(ursi, date)

    if not split:
        ion_tec = [r[1] for r in row_data]
        sat_tec = [r[2] for r in row_data]

        r = round(pearsonr(sat_tec, ion_tec)[0], 2)

        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
        ax.set_xlim(15, 15)

        ax.set_title(
            f"{ursi} lat: {coords['lat']}, long: {coords['lat']},\nr={r_sun}",
            fontsize=15,
        )

        plot_graph(
            ax=ax,
            x_ax=ion_tec,
            y_ax=sat_tec,
            x_label='ion_tec',
            y_label='sat_tec',
            title=f"",
            regression=True,
            const=True,
        )
    else:
        sun_data, moon_data = split_to_sun_moon(row_data, ursi, date)

        ion_tec_sun = [r[1] for r in sun_data]
        ion_tec_moon = [r[1] for r in moon_data]
        sat_tec_sun = [r[2] for r in sun_data]
        sat_tec_moon = [r[2] for r in moon_data]

        r_sun = round(pearsonr(sat_tec_sun, ion_tec_sun)[0], 2)
        r_moon = round(pearsonr(sat_tec_moon, ion_tec_moon)[0], 2)

        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

        ax[0].set_xlim(15, 15)
        ax[1].set_xlim(15, 15)

        ax[0].set_title(
            f"{ursi} Sun lat: {coords['lat']}, long: {coords['lat']},\nr={r_sun}",
            fontsize=15,
        )
        ax[1].set_title(
            f"{ursi} Moon lat: {coords['lat']}, long: {coords['lat']},\nr={r_moon}",
            fontsize=15,
        )

        plot_graph(
            ax=ax[0],
            x_ax=ion_tec_sun,
            y_ax=sat_tec_sun,
            x_label='ion_tec',
            y_label='sat_tec',
            title=f"",
            regression=True,
            const=True,
        )
        plot_graph(
            ax=ax[1],
            x_ax=ion_tec_moon,
            y_ax=sat_tec_moon,
            x_label='ion_tec',
            y_label='sat_tec',
            title=f"",
            regression=True,
            const=True,
        )
