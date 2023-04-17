import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as Ax

from pandas import DataFrame as DF

from plot.utils import make_linear_regression



def set_title_date_k_err(ax: Ax, date: str, k: float, err: float) -> None:
    ax.set_title(
        f"{date}, k={round(k, 3)}, err={round(err, 3)}",
        fontsize=15,
    )


def plot_graph(
        ax: Ax,
        x_ax: list,
        y_ax: list,
        x_label: str,
        y_label: str,
        date: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
) -> Ax:
    if regression:
        reg = make_linear_regression(y_ax, x_ax, const)
        linspace = np.linspace(0, max(x_ax), 100)
        ax.plot(linspace, reg.predict(linspace), c=edgecolor)

        ax.set_title(
            f"{date}, k={round(reg.params[0], 3)}, err={round(reg.bse[0], 3)}",
            fontsize=15,
        )
    else:
        ax.set_title(f"{date}", fontsize=15)
    
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.scatter(x=x_ax, y=y_ax, marker='o', c=color, edgecolor=edgecolor)
    ax.grid()

    return ax


def plot_linear_graph(
        ax: Ax,
        df: DF,
        x_name: str,
        y_name: str,
        x_label: str,
        y_label: str,
        date: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
) -> Ax:
    x_ax = df[x_name]
    y_ax = df[y_name]

    return plot_graph(
            ax, x_ax, y_ax, x_label, y_label, date, color,
            edgecolor, regression, const,
    )


#def plot_squared_graph(ax, df, x_col, y_col, x_label, y_label, date, color='y', edgecolor='g'):    
#    x_ax = df[x_col]
#    y_ax = [v**2 for v in df[y_col]]
#
#    return plot_graph(ax, x_ax, y_ax, x_label, y_label, date, color=color, edgecolor=edgecolor)


#def plot_tec_b0_graph(df, date: str):
#    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
#    
##     ax.set_xlim(None, 30)
##     ax.set_ylim(None, 300)
#     
#    return plot_linear_graph(ax, df, 'tec', 'b0', 'TEC', 'B0', date)
#
#def plot_tec_b0_split_graph(sun, moon, date: str):
#    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
#    
##     ax[0].set_xlim(None, 30)
##     ax[0].set_ylim(None, 300)
##     ax[1].set_xlim(None, 30)
##     ax[1].set_ylim(None, 300)
#    
#    
#    plot_linear_graph(
#        ax[0], sun, 'tec', 'b0', 'TEC', 'B0',
#        'Sun ' + date, color='orange', edgecolor='r',
#    )
#    plot_linear_graph(
#        ax[1], moon, 'tec', 'b0', 'TEC', 'B0',
#        'Moon ' + date, color='purple', edgecolor='b',
#    )
#    
#    
#def subplot_tec_b0_graph(df, date: str, ax) -> None:
#    ax = plot_linear_graph(ax, df, 'tec', 'b0', 'TEC', 'B0', date)
#    
##     ax.set_xlim(None, 30)
##     ax.set_ylim(None, 300)
#    
#    return ax
#
#
#def subplot_tec_b0_split_graph(sun, moon, date: str, ax) -> None:
#    x_sun = sun['tec']
#    y_sun = sun['b0']
#    x_moon = moon['tec']
#    y_moon = moon['b0']
#    
#    reg_sun = make_linear_regression(y_sun, x_sun)
#    reg_moon = make_linear_regression(y_moon, x_moon)
#    
#    ax.set_xlabel('TEC', fontsize=15)
#    ax.set_ylabel('B0', fontsize=15)
#    ax.set_title(
#        f"{date},\n\
#        k_sun={round(reg_sun.params[0], 3)}, err={round(reg_sun.bse[0], 3)}\n\
#        k_moon={round(reg_moon.params[0], 3)}, err={round(reg_moon.bse[0], 3)}",
#        fontsize=15,
#    )
#    
#    
#    ax.scatter(x=x_sun, y=y_sun, marker='o', c='orange', edgecolor='r')
#    ax.plot(np.linspace(0, max(x_sun), 100), reg_sun.predict(np.linspace(0, max(x_sun), 100)), c='r')
#    
#    ax.scatter(x=x_moon, y=y_moon, marker='o', c='purple', edgecolor='b')
#    ax.plot(np.linspace(0, max(x_moon), 100), reg_moon.predict(np.linspace(0, max(x_moon), 100)), c='b')
#    
#    ax.grid()       
##     ax.set_xlim(None, 30)
##     ax.set_ylim(None, 300)
#    
#    return ax
#
#
#def plot_tec_b0_for_day_graph(
#    ursi: str,
#    date: str,
#    ax = None,
#    splitted = False,
#) -> None:
#    df = cast_data_to_dataframe(
#        select_hour_avr_for_day(ursi, date),
#        columns=['hour', 'f0f2', 'tec', 'b0'],
#    )
#    
#    sunrise, sunset = get_sunrise_sunset(date, select_coords_by_ursi(ursi))
#    hour = df['hour']
#    
#    if sunrise < sunset:
#        sun = df[(hour >= sunrise) & (hour < sunset)]
#        moon = df[(hour < sunrise) | (hour >= sunset)]
#    else:
#        sun = df[(hour >= sunrise) | (hour < sunset)]
#        moon = df[(hour < sunrise) & (hour >= sunset)]
#
#
#    if ax != None:
#        if not splitted:
#            subplot_tec_b0_graph(df, date, ax)
#        else:
#            subplot_tec_b0_split_graph(sun, moon, date, ax)
#    else:
#        if not splitted:
#            plot_tec_b0_graph(df, date)
#        else:
#            plot_tec_b0_split_graph(sun, moon, date)
#    
#    
#    
#def plot_tec_b0_for_each_day_in_month_graph(
#    ursi: str,
#    month: int,
#    splitted=False,
#) -> None:
#    coords = select_coords_by_ursi(ursi)
#    
#    fig, ax_list = plt.subplots(ncols=3, nrows=11,figsize=(20, 60))
#    fig.subplots_adjust(
#        left=0.1,
#        bottom=0.1,
#        right=0.9,
#        top=0.9,
#        wspace=0.4,
#        hspace=0.6,
#    )
#    
#    axes = []
#    for ax in ax_list:
#        axes = [*axes, *ax]
#    
#    suptitle = f"{ursi}, lat: {coords['lat']} \
#    long: {coords['long']}, Month: {month}"
#    
#    fig.suptitle(suptitle, fontsize=20, y=0.92)
#    
#    for day in range(1, get_month_days_count(month) + 1):
#        try:
#            str_month = f'0{month}' if month < 10 else f'{month}'
#            str_day = f'0{day}' if day < 10 else f'{day}'
#            plot_tec_b0_for_day_graph(
#                ursi,
#                f"2019-{str_month}-{str_day}",
#                axes[day - 1],
#                splitted,
#            )
#        except:
#            continue
#
