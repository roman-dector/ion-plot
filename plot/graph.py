import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from matplotlib.axis import Axis as Ax
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
        title: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
        moon: bool=False,
) -> Ax:
    if regression:
        if const:
            if moon:
                reg = make_linear_regression(x_ax, y_ax, const)
                linspace = np.linspace(0, max(y_ax), 100)
                ax.plot(reg.predict(sm.add_constant(linspace)), linspace, c=edgecolor)
            else:
                reg = make_linear_regression(y_ax, x_ax, const)
                linspace = np.linspace(0, max(x_ax), 100)
                ax.plot(linspace, reg.predict(sm.add_constant(linspace)), c=edgecolor)

            ax.set_title(
                f"{title}, k={round(reg.params[1], 3)}, k_err={round(reg.bse[1], 3)},\n\
                const={round(reg.params[0], 3)}, const_err={round(reg.bse[0], 3)}",
                fontsize=15,
            )
        else:
            reg = make_linear_regression(y_ax, x_ax, const)
            linspace = np.linspace(0, max(x_ax), 100)
            ax.plot(linspace, reg.predict(linspace), c=edgecolor)

            ax.set_title(
                f"{title}, k={round(reg.params[1], 3)}, k_err={round(reg.bse[1], 3)}",
                fontsize=15,
            )
    else:
        ax.set_title(f"{title}", fontsize=15)
    
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
        title: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
        moon: bool=False,
) -> Ax:
    x_ax = df[x_name]
    y_ax = df[y_name]

    return plot_graph(
            ax, x_ax, y_ax, x_label, y_label, title, color,
            edgecolor, regression, const, moon,
    )
