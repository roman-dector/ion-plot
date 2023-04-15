import numpy as np

from plot.utils import make_linear_regression


def set_title_date_k_err(ax, date, k, err) -> None:
    ax.set_title(
        f"{date}, k={round(k, 3)}, err={round(err, 3)}",
        fontsize=15,
    )


def plot_graph(ax, x_ax, y_ax, x_label, y_label, date, color='y', edgecolor='g'):
    reg = make_linear_regression(y_ax, x_ax)
    linspace = np.linspace(0, max(x_ax), 100)
    
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_title(
        f"{date}, k={round(reg.params[0], 3)}, err={round(reg.bse[0], 3)}",
        fontsize=15,
    )
    ax.scatter(x=x_ax, y=y_ax, marker='o', c=color, edgecolor=edgecolor)
    ax.plot(linspace, reg.predict(linspace), c=edgecolor)
    ax.grid()
    
    return ax


def plot_linear_graph(ax, df, x_col, y_col, x_label, y_label, date, color='y', edgecolor='g'):    
    x_ax = df[x_col]
    y_ax = df[y_col]

    return plot_graph(ax, x_ax, y_ax, x_label, y_label, date, color=color, edgecolor=edgecolor)


def plot_squared_graph(ax, df, x_col, y_col, x_label, y_label, date, color='y', edgecolor='g'):    
    x_ax = df[x_col]
    y_ax = [v**2 for v in df[y_col]]

    return plot_graph(ax, x_ax, y_ax, x_label, y_label, date, color=color, edgecolor=edgecolor)

