import datetime
import numpy as np
import matplotlib.pyplot as plt
from .experiment import Experiment
from scipy.interpolate import UnivariateSpline
from .helper_functions import (
    _get_smoothing,
    _get_cut,
    _x_to_index,
    _xbaseline_to_ibaseline,
)


def plot_package_help():
    print(
        "Example of easy plotting with this package is:\n"
        "fig = fp.figure() <= defaults to 2-column journal image at 150 dpi\n"
        "fp.plot_exp(exp, *file_numbers (written as 1,2,3,... or blank for"
        "all files))\n"
        "fp.plot_tailer(x_label=, y_label=, set_position=, ...)\n"
        "fp.plot_exp(exp, *other_files, new_fig=False, new_ax=False)\n"
        "fp.plot_tailer(legend=True, x_lim=, y_lim=, ...)\n\n"
        "fp.plot_fits(exp, file_numbers)\n"
        "fp.plot_tailer(x_tick_values=, x_tick_sides=, x_label_sides=)\n\n"
        "fp.plot_metadata(exp, *file_numbers, x_regex=, y_regex=, etc.)"
        "\n\n"
        "To produce multiplot figures, the general template is:\n"
        "fig = fp.figure(figsize=(w, h), dpi=n) # where w, h are in inches\n"
        "# ** Create plot on axis in that figure **\n"
        "ax = fig.add_subplot(111, label=unique) # Since you are positioning "
        "the axis yourself, you don't need to worry about the 111. The label "
        "just needs to be a unique number so that different axes don't "
        "interfere with eachother.\n"
        "fp.plot(x, y, you_know_the_drill) # yes, you can use fpack (no "
        "matplotlib.pyplot import required for the basics!)\n"
        "# ** Plot created **\n"
        "fp.plot_tailor(ax=ax, set_position = [x0, y0, w, h]) # where all "
        "are in terms of percentage of figure / 100. So an axis that uses "
        "100% of the x dimension of your figure will have w = 1.\n"
        "fp.plot_savefig(filename, transparent=bool, dpi=n) # where bool "
        "decides if your figure and axes background is visible or not."
    )


def plot_manual_guide():
    print(
        "Here are some of the usual functions used while "
        "making proffesional looking plots in python.\n"
        "This assume 'import matplotlib.pyplot as plt' "
        "has already been entered.\n"
        "plt.figure\n"
        "plt.subplot\n"
        "plt.xlabel\n"
        "plt.ylabel\n"
        "plt.ticklabel_format\n"
        "plt.minorticks_on\n"
        "plt.legend\n"
        "plt.grid\n"
        "plt.plot\n"
        "plt.text\n"
        "ax.set_position\n"
        "plt.savefig\n"
        "plt.gcf\n"
        "plt.gca\n"
        "plt.rc('text', usetex=True) -- use LaTeX to render all text"
        "plt.rc('font', family='serif') -- font best matched to APL\n"
        "plt.rc('text.latex',"
        r"preamble=r'\usepackage{siunitx}, \usepackage{mathptmx}')"
    )


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


def clean_plot_data(x_data, y_data, which_axis=None):
    """Takes in data littered with Nones and returns cleaned data as two
    numpy arrays with the np.float datatype.

    which_axis determines which axis (0 or 1) to check for np.nan values.
    A value of -1 (default) checks and removes np.nan entries from both axes.

    """
    if len(x_data) != len(y_data):
        raise ValueError(
            f"x_data and y_data are not the same length. "
            f"x is {len(x_data)} and y is {len(y_data)}."
        )
    combined = np.dstack((x_data, y_data))[0]
    if which_axis is None:
        combined_cut = combined[np.isfinite(x_data) & np.isfinite(y_data)]
    elif which_axis == 0:
        combined_cut = combined[np.isfinite(x_data)]
    elif which_axis == 1:
        combined_cut = combined[np.isfinite(y_data)]
    else:
        raise ValueError(
            "which_axis needs to be -1, 0, or 1. It is currently {which_axis}."
        )
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
    *args, new_fig=True, new_ax=False, figsize=(6.67, 4), dpi=150, **kwargs,
):
    fig, ax = _new_fig_andor_ax(
        new_fig=new_fig, new_ax=new_ax, figsize=figsize, dpi=dpi
    )
    plt.plot(*args, **kwargs)


# Include the ability to do log plots along various axes.


def _tick_helper(
    x_tick_values=None,
    x_tick_labels=None,
    x_tick_sides=(None, None),
    x_tick_label_sides=(None, None),
    y_tick_values=None,
    y_tick_labels=None,
    y_tick_sides=(None, None),
    y_tick_label_sides=(None, None),
    minor_ticks=True,
    ticks_in_out_inout=(None, None),
    tick_size=None,
    science_ticks=None,
    science_ticks_offset=None,
):
    if minor_ticks is True:
        plt.minorticks_on()
    elif minor_ticks is False:
        plt.minorticks_off()
    if ticks_in_out_inout != (None, None):
        plt.tick_params(
            axis="x",
            direction=(ticks_in_out_inout[0])
        )
        plt.tick_params(
            axis="y",
            direction=(ticks_in_out_inout[1])
        )
    if tick_size is not None:
        plt.tick_params(
            axis="both",
            labelsize=tick_size
        )

    if x_tick_sides != (None, None):
        plt.tick_params(
            bottom=x_tick_sides[0],
            top=x_tick_sides[1]
        )
    if x_tick_label_sides != (None, None):
        plt.tick_params(
            labelbottom=x_tick_label_sides[0],
            labeltop=x_tick_label_sides[1],
        )
    if y_tick_sides != (None, None):
        plt.tick_params(
            left=y_tick_sides[0],
            right=y_tick_sides[1]
        )
    if y_tick_label_sides != (None, None):
        plt.tick_params(
            labelleft=y_tick_label_sides[0],
            labelright=y_tick_label_sides[1],
        )

    if x_tick_values is not None:
        plt.gca().xaxis.set_ticks(x_tick_values)
    if x_tick_labels is not None:
        plt.gca().xaxis.set_ticklabels(x_tick_labels)
    if y_tick_values is not None:
        plt.gca().yaxis.set_ticks(y_tick_values)
    if y_tick_labels is not None:
        plt.gca().yaxis.set_ticklabels(y_tick_labels)

    if science_ticks is not None and x_tick_labels is None:
        plt.ticklabel_format(
            style="sci", scilimits=(-2, 4), axis="both", useOffset=science_ticks_offset
        )
    elif science_ticks is not None and x_tick_labels is not None:
        print(
            "Warning, x_tick_labels cannot be set "
            "at the same time as science_ticks."
        )


def _label_helper(
    x_label=None,
    y_label=None,
    label_positions=(None, None),
    label_size=None,
):
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if label_size is not None:
        x_text = plt.gca().xaxis.get_label().get_text()
        y_text = plt.gca().yaxis.get_label().get_text()
        plt.xlabel(x_text, fontsize=label_size)
        plt.ylabel(y_text, fontsize=label_size)
    if label_positions != (None, None):
        plt.gca().xaxis.set_label_position(label_positions[0])
        plt.gca().yaxis.set_label_position(label_positions[1])


def plot_tailor(
    *,
    ax=None,
    x_label=None,
    y_label=None,
    x_logscale=None,
    y_logscale=None,
    x_lim=(None, None),
    y_lim=(None, None),
    title=None,
    title_size=None,
    legend=None,
    legend_title=None,
    legend_loc=None,
    legend_text=None,
    legend_fontsize=None,
    legend_alpha=None,
    grid=None,
    x_tick_values=None,
    x_tick_labels=None,
    x_tick_sides=(None, None),
    x_tick_label_sides=(None, None),
    y_tick_values=None,
    y_tick_labels=None,
    y_tick_sides=(None, None),
    y_tick_label_sides=(None, None),
    minor_ticks=True,
    ticks_in_out_inout=(None, None),
    label_positions=(None, None),
    label_size=None,
    tick_size=None,
    science_ticks=None,
    science_ticks_offset=None,
    set_position=None,
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
        legend = plt.legend()
        if legend_loc is not None:
            plt.legend(loc=legend_loc)
        if legend_fontsize is not None:
            plt.legend(fontsize=legend_fontsize)
        if legend_alpha is not None:
            plt.legend(framealpha=legend_alpha)
        if legend_title is not None:
            plt.legend(title=legend_title)
        if legend_text is not None:
            legend_texts = zip(legend.get_texts(), legend_text)
            [text_spot.set_text(text) for (text_spot, text) in legend_texts]
    elif legend is False:
        legend = plt.legend()
        legend.remove()

    if grid is not None:
        plt.grid(axis="both", which="both", alpha=0.25)

    if x_logscale is not None:
        plt.xscale("log")
    if y_logscale is not None:
        plt.yscale("log")

    _tick_helper(
        x_tick_values=x_tick_values,
        x_tick_labels=x_tick_labels,
        x_tick_sides=x_tick_sides,
        x_tick_label_sides=x_tick_label_sides,
        y_tick_values=y_tick_values,
        y_tick_labels=y_tick_labels,
        y_tick_sides=y_tick_sides,
        y_tick_label_sides=y_tick_label_sides,
        minor_ticks=minor_ticks,
        ticks_in_out_inout=ticks_in_out_inout,
        tick_size=tick_size,
        science_ticks=science_ticks,
        science_ticks_offset=science_ticks_offset
    )

    _label_helper(
        x_label=x_label,
        y_label=y_label,
        label_positions=label_positions,
        label_size=label_size,
    )

    if title is not None:
        plt.title(title, fontsize=title_size)
    if set_position is not None:
        plt.gca().set_position([*set_position])
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
    elif x_value is not None:
        i = _x_to_index(x_data, x_value)
        plt.axvline(x_data[i])
        return i



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
# plot_exp. Derivative and integral plotting by data and by spline.


def plot_exp(
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
    ibaseline=(None, None),
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
    if xbaseline != (None, None) and ibaseline != (None, None):
        raise ValueError(
            f"Please only specify either xbaseline OR ibaseline.\n"
            f"Currently \nxbaseline = {xbaseline}\nibaseline = {ibaseline}"
        )
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
            if y_data is None:
                continue
        elif guess:
            x_data, y_data = exp.get_xy_guess(
                file_number, x_column=x_column, x_density=x_density
            )
            if y_data is None:
                continue
        else:
            x_data, y_data = exp.get_xy_data(
                file_number, x_column=x_column, y_column=y_column
            )
        if integrate:
            if ibaseline == (None, None):
                i_baseline = _xbaseline_to_ibaseline(x_data, xbaseline)
            y_data = np.cumsum(
                y_data - np.average(y_data[i_baseline[0] : i_baseline[1]])
            )

        if waterfall:
            y_data = y_data - y_data[0] + n * waterfall

        if metadata_label is not None:
            label = exp.get_metadata(
                file_number,
                regex=metadata_label,
                repl=repl,
                match_number=match_number,
            )
        else:
            label = f"File: {file_number}"

        ax.plot(
            x_data[::skip_points],
            y_data[::skip_points],
            fmt,
            label=label,
            **plot_kw,
        )

    if metadata_title is not None:
        title = exp.get_metadata(
            file_numbers[0],
            regex=metadata_title,
            repl=repl,
            match_number=match_number,
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
    plot_exp(
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
    plot_exp(
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
    plot_exp(
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
        plot_exp(
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

        plot_exp(
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
        print(f"No fit data found for file {file_number}\n{e}")

    if auto:
        x_column, y_column = exp.check_xy_columns(x_column, y_column)
        x_data, y_data = exp.get_xy_data(
            file_number, x_column=x_column, y_column=y_column
        )
        smoothing = _get_smoothing(y_data, smoothing)
        cut_range = _get_cut(x_data, y_data, xbaseline, cut_scale)
        y_cut = np.copy(y_data)
        y_cut[(y_data >= cut_range[0]) & (y_data <= cut_range[1])] = y_data[0]
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
            x_data[ibaseline[0] : ibaseline[1]],
            y_data[ibaseline[0] : ibaseline[1]],
            "C4--",
            alpha=0.5,
            label="Baseline Data",
        )
        plt.autoscale(enable=False, axis="y")
        if derivative:
            plt.plot(
                x_data[ifit_range[0] : ifit_range[1]],
                deriv(x_data[ifit_range[0] : ifit_range[1]]),
                "C2:",
                label="Spline",
            )
        else:
            plt.plot(
                x_data[ifit_range[0] : ifit_range[1]],
                integral(x_data[ifit_range[0] : ifit_range[1]]),
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
    which_axis=None,
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
    x_data, y_data = clean_plot_data(x_data, y_data, which_axis=which_axis)

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

    x_data1, y_data1 = exp.get_xy_data(
        file_1, x_column=x_column_1, y_column=y_column_1
    )  # noqa
    x_data2, y_data2 = exp.get_xy_data(
        file_2, x_column=x_column_2, y_column=y_column_2
    )  # noqa

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


def create_figure_grid(
    num_x, num_y, alpha=0.3, bbox=(0, 0, 1, 1), *, fig=None
):  # noqa
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
