from .experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import datetime
from .extractors import _get_smoothing, \
    _get_cut, \
    _x_to_index, \
    _xbaseline_to_ibaseline


def show():
    """Just a wrapper for plt.show()
    """
    plt.show()


def plot_savefig(filename, transparent=True, dpi=600, **kwargs):
    """Uses plt.savefig with better defaults.

    transparent = True : No white background behind figure.
    dpi = 600          : Recommended minimum dpi for published graphics.
    """
    plt.savefig(filename, transparent=transparent, dpi=dpi, **kwargs)


def clean_plot_data(x_data, y_data):
    """Takes in data littered with Nones and returns cleaned data as two
    numpy arrays with the np.float datatype.

    """
    if len(x_data) != len(y_data):
        raise ValueError(
            f"x_data and y_data are not the same length. "
            f"x is {len(x_data)} and y is {len(y_data)}."
        )
    combined = np.dstack((x_data, y_data))[0]
    combined_cut = combined[np.isfinite(x_data) & np.isfinite(y_data)]
    return combined_cut[:, 0], combined_cut[:, 1]


def figure(*, column_width=2, height=4, figsize=None, dpi=150) -> plt.Figure:
    """Create a standard figure size for paper submission. All units in
    inches.

    """
    if figsize is not None:
        try:
            width = figsize[0]
            height = figsize[1]
        except IndexError:
            raise IndexError("figsize must be figsize=(w,h)")
    elif (column_width == 1) or (column_width == 2):
        width = 3.37 * (column_width == 1) + 6.69 * (column_width == 2)
    else:
        raise ValueError(
            "Set 'column_width' to 1 or 2 to get "
            "common widths for journal publication "
            "or set figsize=(w,h) for custom sizes."
        )

    fig = plt.figure(figsize=(width, height), dpi=dpi)

    return fig


def plot(
    *args,
    new_fig=True,
    new_ax=False,
    figsize=(6.67, 4),
    dpi=150,
    **kwargs,
):
    fig, ax = _new_fig_andor_ax(
        new_fig=new_fig, new_ax=new_ax, figsize=figsize, dpi=dpi
    )
    plt.plot(*args, **kwargs)


# Include the ability to do log plots along various axes.

def plot_tailor(
    *,
    ax=None,
    x_label=None,
    y_label=None,
    x_lim=(None, None),
    y_lim=(None, None),
    title=None,
    title_size=12,
    legend=None,
    legend_title="",
    legend_loc=0,
    legend_text=None,
    legend_size=8,
    legend_title_size=10,
    legend_alpha=1,
    grid=False,
    x_tick_values=None,
    x_tick_labels=None,
    x_tick_sides=(True, False),
    x_tick_label_sides=(True, False),
    y_tick_values=None,
    y_tick_labels=None,
    y_tick_sides=(True, False),
    y_tick_label_sides=(True, False),
    minor_ticks=True,
    ticks_in_out_inout=("out", "out"),
    label_positions=("bottom", "left"),
    label_size=10,
    tick_size=8,
    set_position=[0.1, 0.1, 0.88, 0.88],
):
    """Performs a set of standard axes, labels, legends, and ticks
    manipulations to quickly and easily make plots look more
    professional and standard.
    """
    if ax:
        try:
            plt.sca(ax)
        except ValueError:
            print(f"Failed to set current ax to {ax}")

    if legend is True:
        legend = plt.legend(
            loc=legend_loc,
            fontsize=legend_size,
            # title_fontsize=legend_title_size,
            framealpha=legend_alpha,
            title=legend_title,
        )
        if legend_text:
            legend_texts = zip(legend.get_texts(), legend_text)
            [text_spot.set_text(text) for (text_spot, text) in legend_texts]
    elif legend is False:
        legend = plt.legend()
        legend.remove()

    if grid:
        plt.grid(axis="both", which="both", alpha=0.25)

    if minor_ticks:
        plt.minorticks_on()
    plt.tick_params(
        axis="x",
        direction=(ticks_in_out_inout[0]),
        which="both",
        bottom=x_tick_sides[0],
        top=x_tick_sides[1],
        labelbottom=x_tick_label_sides[0],
        labeltop=x_tick_label_sides[1],
        labelsize=tick_size,
    )
    plt.tick_params(
        axis="y",
        direction=(ticks_in_out_inout[1]),
        which="both",
        left=y_tick_sides[0],
        right=y_tick_sides[1],
        labelleft=y_tick_label_sides[0],
        labelright=y_tick_label_sides[1],
        labelsize=tick_size,
    )
    if x_tick_values:
        plt.gca().xaxis.set_ticks(x_tick_values)
    if x_tick_labels:
        plt.gca().xaxis.set_ticklabels(x_tick_labels)
    if y_tick_values:
        plt.gca().yaxis.set_ticks(y_tick_values)
    if y_tick_labels:
        plt.gca().yaxis.set_ticklabels(y_tick_labels)

    plt.ticklabel_format(style="sci", scilimits=(-2, 4), axis="both", useOffset=False)  # noqa
    if x_label:
        plt.xlabel(x_label, fontsize=label_size)
    if y_label:
        plt.ylabel(y_label, fontsize=label_size)
    plt.gca().xaxis.set_label_position(label_positions[0])
    plt.gca().yaxis.set_label_position(label_positions[1])
    plt.gca().set_position([*set_position])
    if title:
        plt.title(title, fontsize=title_size)

    plt.xlim(*x_lim)
    plt.ylim(*y_lim)


def get_index(data, x_value, row=0):
    """Accepts a dataset organized as [[x_data], [y_data_1], ... , [y_data_n]].
    Returns the index of the 'row'-th dataset that is closest to 'x_value'.
    """
    try:
        search_data = data[row]
    except IndexError:
        search_data = data
    return _x_to_index(search_data, x_value)


def plot_xvline(exp: Experiment, point_number=None, x_value=None):
    x_data, _ = exp.get_xy_data(0)
    if point_number:
        plt.axvline(x_data[point_number])
        return x_data[point_number]
    elif not (x_value is None):
        i = _x_to_index(x_data, x_value)
        plt.axvline(x_data[i])
        return i


def plot_package_help():
    print(
        "Example of easy plotting with this package is:\n"
        "fig = fp.figure() <= defaults to 2-column journal image at 150 dpi\n"
        "fp.plot_scans(exp, *file_numbers (written as 1,2,3,... or blank for"
        "all files))\n"
        "fp.plot_tailer(x_label=, y_label=, set_position=, ...)\n"
        "fp.plot_scans(exp, *other_files, new_fig=False, new_ax=False)\n"
        "fp.plot_tailer(legend=True, x_lim=, y_lim=, ...)\n\n"
        "fp.plot_fits(exp, file_numbers)\n"
        "fp.plot_tailer(x_tick_values=, x_tick_sides=, x_label_sides=)\n\n"
        "fp.plot_metadata(exp, *file_numbers, x_regex=, y_regex=, etc.)")


def plot_manual_guide():
    print(
        "Here are some of the usual functions used while "
        "making proffesional looking plots in python.\n"
        "This assume 'import matplotlib.pyplot as plt' "
        "has already been entered.\n"
        "fig = plt.figure(figsize=(x, y), dpi=DPI)\n"
        "ax = plt.subplot(111, sharex=False, sharey=False)\n"
        "plt.xlabel('XLabel', fontsize=20)\n"
        "plt.ylabel('YLabel', fontsize=20)\n"
        "plt.ticklabel_format(style='sci', scilimits=(-1,3),"
        " axis='both', useOffset=False)\n"
        "plt.minorticks_on()\n"
        "plt.legend(loc=0, fontsize=13, title_fontsize=15,"
        "framealpha=1, title='Title')\n"
        "plt.grid(axis='both', which='both', alpha=0.25)\n"
        "plt.plot(x_list, y_list, 'bo-', linewidth=1, markersize=1, alpha=1)\n"
        "plt.text(x, y, 'text', fontsize=14)\n"
        "ax.set_position([x, y, w, h]) -- allows manual placement"
        " of axis within its figure\n"
        "plt.savefig('name', transparent=True, dpi=DPI)\n"
        "plt.gcf() -- get current figure || "
        "plt.gca() -- get current axis\n"
        "fig.patches.extend([plt.Rectangle((x, y), w, h,"
        " fill=True, color='w', alpha=1, zorder=1000,"
        " transform=fig.transFigure, figure=fig)])\n"
        "plt.rc('text', usetex=True) -- use LaTeX to render all text"
        "plt.rc('font', family='serif') -- font best matched to APL\n"
        "plt.rc('text.latex',"
        r"preamble=r'\usepackage{siunitx}, \usepackage{mathptmx}')")


def _new_fig_andor_ax(new_fig=True, new_ax=False, figsize=(6.69, 4), dpi=150):
    if new_fig is True:
        fig = figure(figsize=figsize)
        ax = fig.add_subplot(111, label=str(datetime.datetime.now()))
    else:
        fig = plt.gcf()
        if new_ax is True:
            ax = fig.add_subplot(111, label=str(datetime.datetime.now()))
        else:
            ax = plt.gca()
    return fig, ax


# Add plotting with waterfall offset labelled by some extracted number
# - voltage for example [0, 1, 5, 10, 20]. Wrap heatmat into
# plot_scans. Derivative and integral plotting by data and by spline.

def plot_scans(
    exp: Experiment,
    *file_numbers,
    new_fig=True,
    new_ax=False,
    figsize=(6.69, 4),
    dpi=150,
    waterfall=False,
    color="",
    x_column=None,
    y_column=None,
    integrate=False,
    xbaseline=(None, None),
    fit=False,
    guess=False,
    fmt="",
    x_density=1,
    skip_points=1,
    metadata_label=None,
    metadata_title=None,
    repl=r"\1",
    match_number=0,
    **plot_kw,
):
    """General purpose plotting function to plot several plots (or one)
    all on one figure.

    fit=True to plot the fit for a set of scans
    """
    ax: plt.Axes
    fig, ax = _new_fig_andor_ax(
        new_fig=new_fig, new_ax=new_ax, figsize=figsize, dpi=dpi
    )

    file_numbers = exp.check_file_numbers(file_numbers)
    x_column, y_column = exp.check_xy_columns(x_column, y_column)
    for n, file_number in enumerate(file_numbers):
        if fit:
            x_data, y_data = exp.get_xy_fit(
                file_number, x_column=x_column, x_density=x_density
            )
        elif guess:
            x_data, y_data = exp.get_xy_guess(
                file_number, x_column=x_column, x_density=x_density
            )
        else:
            x_data, y_data = exp.get_xy_data(
                file_number, x_column=x_column, y_column=y_column
            )
        if integrate:
            i_baseline = _xbaseline_to_ibaseline(x_data, xbaseline)
            y_data = np.cumsum(
                y_data - np.average(y_data[i_baseline[0]: i_baseline[1]])
            )

        if waterfall:
            y_data = y_data - y_data[0] + n * waterfall

        if not (metadata_label is None):
            label = exp.get_metadata(
                file_number, regex=metadata_label, repl=repl,
                match_number=match_number
            )
        else:
            label = f"File: {file_number}"

        ax.plot(
            x_data[::skip_points], y_data[::skip_points], fmt, label=label,
            **plot_kw
        )

    if not (metadata_title is None):
        title = exp.get_metadata(
            file_number, regex=metadata_title, repl=repl,
            match_number=match_number
        )
    else:
        title = None

    axes_labels = exp.get_scan(file_numbers[0]).axes
    x_label, y_label = axes_labels[x_column], axes_labels[y_column]
    plot_tailor(x_label=x_label, y_label=y_label, title=title)


def plot_fits(
    exp: Experiment,
    *file_numbers,
    new_fig=True,
    new_ax=False,
    figsize=(6.69, 4),
    dpi=150,
    waterfall=False,
    x_column=None,
    y_column=None,
    x_density=1,
):
    plot_scans(
        exp,
        *file_numbers,
        new_fig=new_fig,
        new_ax=new_ax,
        figsize=figsize,
        waterfall=waterfall,
        dpi=dpi,
        fmt="C0-",
        x_column=x_column,
        y_column=y_column,
        linewidth=1,
    )
    plot_scans(
        exp,
        *file_numbers,
        new_fig=False,
        new_ax=False,
        figsize=figsize,
        waterfall=waterfall,
        dpi=dpi,
        fmt="C3--",
        x_column=x_column,
        fit=True,
        x_density=x_density,
        linewidth=1,
    )


def plot_guess_and_fit(
    exp: Experiment,
    file_number,
    new_fig=True,
    new_ax=False,
    figsize=(6.69, 4),
    dpi=150,
    waterfall=False,
    x_column=None,
    y_column=None,
    x_density=1,
    xbaseline=(None, None),
    cut_scale=1,
    smoothing=1,
    derivative=True,
    xfit_range=(None, None),
    auto=False,
):
    plot_scans(
        exp,
        file_number,
        new_fig=new_fig,
        new_ax=new_ax,
        figsize=figsize,
        waterfall=waterfall,
        dpi=dpi,
        fmt="C0-",
        x_column=x_column,
        y_column=y_column,
        linewidth=1,
    )
    try:
        plot_scans(
            exp,
            file_number,
            new_fig=False,
            new_ax=False,
            figsize=figsize,
            waterfall=waterfall,
            dpi=dpi,
            fmt="C3-",
            x_column=x_column,
            fit=True,
            x_density=x_density,
            linewidth=1,
        )
        plt.gca().get_lines()[-1].set_zorder(10)

        plot_scans(
            exp,
            file_number,
            new_fig=False,
            new_ax=False,
            figsize=figsize,
            waterfall=waterfall,
            dpi=dpi,
            fmt="C1-",
            x_column=x_column,
            guess=True,
            x_density=x_density,
            linewidth=1,
        )
    except TypeError as e:
        print(f"No fit data found\n{e}")

    if auto:
        x_column, y_column = exp.check_xy_columns(x_column, y_column)
        x_data, y_data = exp.get_xy_data(
            file_number, x_column=x_column, y_column=y_column
        )
        smoothing = _get_smoothing(y_data, smoothing)
        cut_range = _get_cut(x_data, y_data, xbaseline, cut_scale)
        y_cut = np.copy(y_data)
        y_cut[(y_data >= cut_range[0]) & (y_data <= cut_range[1])] = 0
        cut_data = np.vstack((x_data, y_cut))
        ibaseline = _xbaseline_to_ibaseline(x_data, xbaseline)
        ifit_range = _xbaseline_to_ibaseline(x_data, xfit_range)
        if not derivative:
            integral = UnivariateSpline(cut_data[0], cut_data[1], k=5)
            integral.set_smoothing_factor(smoothing)
            deriv = integral.derivative()
        else:
            deriv = UnivariateSpline(cut_data[0], cut_data[1], k=4)
            deriv.set_smoothing_factor(smoothing)
        plt.axhspan(
            cut_range[0],
            cut_range[1],
            color="k",
            alpha=0.3,
            label="Ignored Region",
        )
        plt.plot(
            x_data[ibaseline[0]: ibaseline[1]],
            y_data[ibaseline[0]: ibaseline[1]],
            "C4--",
            alpha=0.5,
            label="Baseline Data",
        )
        plt.autoscale(enable=False, axis="y")
        if derivative:
            plt.plot(
                x_data[ifit_range[0]: ifit_range[1]],
                deriv(x_data[ifit_range[0]: ifit_range[1]]),
                "C2:",
                label="Spline",
            )
        else:
            plt.plot(
                x_data[ifit_range[0]: ifit_range[1]],
                integral(x_data[ifit_range[0]: ifit_range[1]]),
                "C2:",
                label="Spline",
            )

    plot_tailor(
        legend=True,
        legend_text=[
            "Data",
            "Fit",
            "Guess",
            "Baseline Data",
            "Spline",
            "Ignored Region",
        ],
    )


def plot_metadata(
    exp: Experiment,
    *file_numbers,
    x_regex=r".*?(-?\d+(?:,\d+)*(?:e\d+)?(?:\.\d+(?:e\d+)?)?).*$",
    x_repl=r"\1",
    x_match_number=0,
    x_fit_param_indexes=None,
    x_return_file_numbers=None,
    y_regex=r".*?(-?\d+(?:,\d+)*(?:e\d+)?(?:\.\d+(?:e\d+)?)?).*$",
    y_repl=r"\1",
    y_match_number=0,
    y_fit_param_indexes=None,
    y_return_file_numbers=None,
    x_column=None,
    y_column=None,
    new_fig=True,
    new_ax=False,
    figsize=(6.67, 4),
    dpi=150,
    fmt="",
    **plot_kw,
):
    """Plots items found from fits, filename, file_number, and info (scan
    parameters found at start of file) against eachother.

    """
    x_data = exp.get_metadata(
        *file_numbers,
        regex=x_regex,
        repl=x_repl,
        match_number=x_match_number,
        fit_param_indexes=x_fit_param_indexes,
        return_file_numbers=x_return_file_numbers,
    )
    y_data = exp.get_metadata(
        *file_numbers,
        regex=y_regex,
        repl=y_repl,
        match_number=y_match_number,
        fit_param_indexes=y_fit_param_indexes,
        return_file_numbers=y_return_file_numbers,
    )
    x_data, y_data = clean_plot_data(x_data, y_data)

    fig, ax = _new_fig_andor_ax(
        new_fig=new_fig, new_ax=new_ax, figsize=figsize, dpi=dpi
    )
    ax.plot(x_data, y_data, fmt, **plot_kw)


def plot_heatmap(
    exp: Experiment,
    *file_numbers,
    new_fig=True,
    new_ax=False,
    figsize=(6.69, 4),
    dpi=150,
    x_column=None,
    y_column=None,
    y_metrics=[0, 0, 0],
    y_scale=1,
    y_offset=0,
    y_data=None,
    cmap=None,
    cbar_label="cbar_label",
    **plot_option_kwargs,
):
    fig, ax = _new_fig_andor_ax(
        new_fig=new_fig, new_ax=new_ax, figsize=figsize, dpi=dpi
    )

    file_numbers = exp.check_file_numbers(file_numbers)
    x_column, y_column = exp.check_xy_columns(x_column, y_column)

    x_data, _ = exp.get_xy_data(file_numbers[0], x_column=x_column)
    if y_data is None:
        y_data = np.arange(
            y_offset, y_offset + y_scale * (len(file_numbers) + 1), y_scale
        )
        z_data = np.zeros((len(file_numbers), len(x_data)))

    for n, file_number in enumerate(file_numbers):
        _, z_data[n] = exp.get_xy_data(file_number, y_column=y_column)

    c = ax.pcolor(x_data, y_data, z_data, cmap=cmap)
    cbar = plt.gcf().colorbar(c, ax=ax)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=8)
    cbar.ax.set_ylabel(cbar_label, fontsize=10, rotation=270)

    plot_options = {
        "x_label": exp.get_scan(file_number).axes[x_column],
        "legend": False,
        "set_position": [0.1, 0.1, 0.7, 0.85],
    }
    plot_options.update(plot_option_kwargs)

    plot_tailor(**plot_options)


def plot_two_axes(
    exp: Experiment,
    file_1,
    file_2,
    x_column_1=None,
    y_column_1=None,
    x_column_2=None,
    y_column_2=None,
):
    x_column_1, y_column_1 = exp.check_xy_columns(x_column_1, y_column_1)
    x_column_2, y_column_2 = exp.check_xy_columns(x_column_2, y_column_2)

    x_data1, y_data1 = exp.get_xy_data(file_1, x_column=x_column_1, y_column=y_column_1)  # noqa
    x_data2, y_data2 = exp.get_xy_data(file_2, x_column=x_column_2, y_column=y_column_2)  # noqa

    fig = figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x_data1, y_data1, label="File " + str(file_1))
    plot_tailor(
        legend_loc=2,
        x_label=exp.get_scan[file_1].axes[x_column_1],
        y_label=exp.get_scan[file_1].axes[y_column_1],
        set_position=[0.1, 0.1, 0.8, 0.88],
    )

    ax2 = fig.add_subplot(212)
    ax2.patch.set_alpha(0)
    ax2.plot(x_data2, y_data2, "r", label="File " + str(file_2))
    plot_tailor(
        legend_loc=1,
        x_label=exp.get_scan(file_2).axes[x_column_2],
        y_label=exp.get_scan(file_2).axes[y_column_2],
        label_position=("top", "right"),
        tick_positions=(True, False, False, True),
        tick_labels=(True, False, False, True),
        set_position=[0.1, 0.1, 0.8, 0.88],
    )


# Needs to remove right axis from first plot, only as y-axis to right
# side. Just one x axis - no plotting second x axis.

def plot_add_y(x_data, y_data, **plot_kw):
    prev_ax = plt.gca()
    ax = plt.gcf().add_subplot(111, label=str(datetime.datetime.now()))
    ax.patch.set_alpha(0)
    ax.plot(x_data, y_data, **plot_kw)
    plot_tailor(set_position=prev_ax.get_position())


def plot_normalized(
    exp: Experiment,
    *file_numbers,
    figsize=(6.69, 4),
    x_column=None,
    y_column=None,
    **kwargs,
):
    fig = figure(figsize=figsize)
    ax = fig.add_subplot(111, label=str(datetime.datetime.now()))

    file_numbers = exp.check_file_numbers(file_numbers)
    x_column, y_column = exp.check_xy_columns(x_column, y_column)

    for file_number in file_numbers:
        x_data, y_data = exp.get_xy_data(
            file_number, x_column=x_column, y_column=y_column
        )
        y_scaled = normalize(y_data)

        ax.plot(x_data, y_scaled, label=f"File: {file_number}")
        plot_tailor(
            x_label=exp.get_scan(file_number).axes[x_column],
            y_label=exp.get_scan(file_number).axes[y_column],
        )


def normalize(y_data):
    y_max = np.max(y_data)
    y_min = np.min(y_data)
    m = 2 / (y_max - y_min)
    b = -(y_max + y_min) / (y_max - y_min)
    return m * y_data + b


def create_figure_grid(num_x, num_y, alpha=0.3, bbox=(0, 0, 1, 1), *, fig=None):  # noqa
    if fig is None:
        fig = plt.gcf()
    x_loc = np.linspace(bbox[0], bbox[2], num_x)
    y_loc = np.linspace(bbox[1], bbox[3], num_y)
    overview_plot = fig.add_subplot(111)
    overview_plot.set_position([0, 0, 1, 1])
    overview_plot.patch.set_alpha(0)
    for x in x_loc:
        overview_plot.axvline(x, alpha=alpha, linewidth=0.4)
    for y in y_loc:
        overview_plot.axhline(y, alpha=alpha, linewidth=0.4)
