"""fitting_core contains functions to automatically fit lorentzians
and provide easy ways to visualize the fitting process.

"""

import numpy as np
from .experiment import Experiment
from .lorentz_functions import (
    absorption_dispersion_mixed,
    absorption_dispersion_derivative_mixed,
)
from .plotting_core import plot_guess_and_fit, _xbaseline_to_ibaseline
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from .helper_functions import _get_smoothing, _get_cut


def fit_package_help():
    print(
        "fit_exp(exp, func, *filenums) will fit various files "
        "to func and place the fit info within the experiment."
        " All fit info can be extracted via exp.get_metadata and"
        "exp.get_xy_fit and exp.get_all_fit_params. Fit can be "
        "easily plotted via fp.plot_scans(exp, *filenums, fit=True).\n"
        "fit_fmr_exp(exp, *filenums, auto=n, derivative=bool) will fit "
        "filenums to n lorentzians automatically (or manually if params)"
        " is specified.\n"
        "fit_and_plot_fmr(exp, filenum, auto=n, ...) will fit filenum to"
        " n lorentzians and plot various automated fitting constructs to"
        " help with troubleshooting automated fitting issues. Play with "
        "the kwargs (shown via fit_and_plot_fmr?) to get a feel for how"
        " the fit may be failing and how to fix it.\n"
        "\n"
        "General recipe for fitting:\n"
        "from scipy.optimize import curve_fit\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "def fit_func(x, param0, param1, ...):\n"
        "    return x * param0 + ... # your function definition to fit\n"
        "params, cov = curve_fit(fit_func, x, y, ...) # get ... via curve_fit?"
        "\nfp.plot(x, y) # plot the data you fit\n"
        "x_fit = np.linspace(x[0], x[-1], n) # n is number of points you want"
        " in x_fit\n"
        "fp.plot(x_fit, func(x_fit, *params)) # it is import x_fit is a numpy"
        " array.\n"
        "# This next part plots the stderr of your fit as a shaded region.\n"
        "stderr = np.sqrt(np.diag(cov))\n"
        "plt.fill_between(x_fit, func(x_fit, *(params - stderr)), "
        "func(x_fit, *(params + stderr)), alpha=0.3, color='k')\n"
        "\n"
        "More advanced fitting will require extracting data or fit parameters"
        " via exp.get_all_fit_params() or exp.get_metadata"
        "(*files, fit_param_indexes=[p1, p2, p3]) where pn are the positions"
        " of the desired fit parameters.\n"
        "Then post-processing and plotting is left entirely to the user."
    )


def _get_auto_params(
    x,
    y,
    derivative=True,
    cut_scale=None,
    xbaseline=(None, None),
    smoothing=None,
):
    smoothing = _get_smoothing(y, smoothing)
    cut_range = _get_cut(x, y, xbaseline, cut_scale)
    y_cut = np.copy(y)
    y_cut[(y >= cut_range[0]) & (y <= cut_range[1])] = 0
    cut_data = np.vstack((x, y_cut))
    if not derivative:
        integral = UnivariateSpline(cut_data[0], cut_data[1], k=5)
        integral.set_smoothing_factor(smoothing)
        deriv = integral.derivative()
    else:
        deriv = UnivariateSpline(cut_data[0], cut_data[1], k=4)
        deriv.set_smoothing_factor(smoothing)
    deriv2 = deriv.derivative()
    x_roots = deriv2.roots()
    derivs = [deriv2.derivatives(x_root)[1] for x_root in x_roots]
    try:
        x_d = np.dstack((x_roots, derivs))[0][(derivs[0] > 0) :]
    except Exception as e:
        print("Roots of spline: ", x_roots,
              "\nRoots of derivative spline: ", derivs)
        if x_roots == [] and derivs == []:
            print("Try decreasing cut_scale and/or smoothing.")
        raise e
    lorentzian_guesses = [
        (y1 - y0, (deriv(x0), 0, x1 - x0, (x1 + x0) / 2))
        for (x0, y0, x1, y1) in zip(*[np.nditer(x_d)] * 4)
    ]
    return_guesses = [
        guess for (mag, guess) in sorted(lorentzian_guesses, reverse=True)
    ]
    return return_guesses


def fit_exp(
    exp: Experiment,
    fit_func,
    *file_numbers,
    x_column=None,
    y_column=None,
    param_guess_func=None,
    param_guess_func_params=None,
    **curve_fit_kwargs,
):
    """Allows for general fitting to *fit_func* across an entire Experiment.
    Currently uses scipy.curve_fit to fit specified data to the supplied
    fit function. Typically, providing the *params* kwargs will be required
    to get good fit results to anything more complex than linear fits.

    For advanced use, specify param_guess_func in the form:
    def guess_func(x, y, extra_param1, extra_param2):
        ** body **
        return [guess_param0, guess_param1, ...]
    To automate the initial parameter procedure for an arbitrary fit function.
    Insert param_guess_func_params as a dictionary of values to pass, unpacked,
    to you param_guess_func function.
    """
    file_numbers = exp.check_file_numbers(file_numbers)
    x_column, y_column = exp.check_xy_columns(x_column, y_column)

    for n in file_numbers:
        x_data, y_data = exp.get_xy_data(
            n, x_column=x_column, y_column=y_column
        )
        if param_guess_func is not None:
            params = param_guess_func(
                x_data, y_data, **param_guess_func_params
            )
            fit_kwargs = {
                "p0": params,
            }
            fit_kwargs.update(curve_fit_kwargs)

        popt, pcov = curve_fit(fit_func, x_data, y_data, **fit_kwargs)
        exp.get_scan(n).set_scan_params(
            fit_func=fit_func, fit_params=popt, fit_covariance=pcov,
        )


def fit_fmr_exp(
    exp: Experiment,
    *file_numbers,
    params=None,
    offset=None,
    auto=2,
    x_column=None,
    y_column=None,
    derivative=True,
    bounds=(-np.inf, np.inf),
    xbaseline=(None, None),
    cut_scale=None,
    smoothing=None,
    absolute_sigma=False,
    xfit_range=(None, None),
    **curve_fit_kwargs,
):
    """Fits curves to symmetric and antisymmetric components of a lorentzian
    [derivative] signal. Detects and fits automatically to any number of
    lorentzians by using the 'auto' kwarg. Set auto=# of lorentzians you expect.
    i.e. auto=2 will fit to a maximum of 2 lorentzians. For manually fitting,
    set auto=False (the default) and specify '*params' in the form [p1, p2, p3,
    p4], [p5, p6, p7, p8], ... where inital guess parameters for each lorentzian
    are organized as [absorption_amplitude, dispersion_amplitude, linewidth,
    position]. Default behavior is to fit to lorentzian derivates; set
    derivatives=False to fit to absorption lineshapes.

    For auto fitting:
    Without any extra kwargs, the function will attempt to automatically
    generate a spline and ignore background data. If fits are failing,
    modify the following *kwargs* accordingly.
    *xbaseline* (x1, x2) sets the x-values that contain your constant
    background.
    *cut_scale* (1) sets the height of the area that is ignored by fitting.
    With sensible xbaseline,
    *smoothing* (1) sets how much noise in your signal is rejected when
    generating a spline. If guesses are sharp, increase smoothing, if
    fitting results in no fits, decrease smoothing. Change by 10x.
    *xfit_range* (x1, x2) sets the x-values that contain the lorentzian(s) you
    want to fit.
    """  # noqa
    file_numbers = exp.check_file_numbers(file_numbers)
    for n in file_numbers:
        x_column, y_column = exp.check_xy_columns(x_column, y_column)
        x_data, y_data = exp.get_xy_data(
            n, x_column=x_column, y_column=y_column
        )
        ifit_range = _xbaseline_to_ibaseline(x_data, xfit_range)
        if auto:
            params = _get_auto_params(
                x_data,
                y_data,
                xbaseline=xbaseline,
                cut_scale=cut_scale,
                smoothing=smoothing,
            )
            if len(params) > auto:
                num_params = auto
                params = params[:auto]
            else:
                num_params = len(params)
        else:
            num_params = len(params)

        if offset is None:
            offset = y_data[0]

        def fit_function(x, offset1, *params1):
            if derivative:
                return offset1 + sum(
                    [
                        absorption_dispersion_derivative_mixed(
                            x, *params1[(4 * n) : 4 * (n + 1)]
                        )
                        for n in range(num_params)
                    ]
                )
            else:
                return offset1 + sum(
                    [
                        absorption_dispersion_mixed(
                            x, *params1[(4 * n) : 4 * (n + 1)]
                        )  # noqa
                        for n in range(num_params)
                    ]
                )

        if xbaseline != (None, None):
            ibaseline = _xbaseline_to_ibaseline(x_data, xbaseline)
            stddev = np.std(y_data[ibaseline[0] : ibaseline[1]])
            err_list = stddev * np.ones(
                len(y_data[ifit_range[0] : ifit_range[1]])
            )
            absolute_sigma = True
        else:
            err_list = None

        params = [item for sublist in params for item in sublist]
        params.insert(0, offset)

        fit_kwargs = {
            "p0": params,
            "bounds": bounds,
            "sigma": err_list,
            "absolute_sigma": absolute_sigma,
        }
        fit_kwargs.update(curve_fit_kwargs)

        x_data_tofit = x_data[ifit_range[0] : ifit_range[1]]
        y_data_tofit = y_data[ifit_range[0] : ifit_range[1]]
        try:
            fit_para, fit_cov = curve_fit(
                fit_function, x_data_tofit, y_data_tofit, **fit_kwargs
            )
        except RuntimeError as e:
            print(f"Failed to fit file number: {n}")
            print(repr(e))
            fit_para = [
                0,
            ]
            fit_cov = [
                0,
            ]

        exp.set_scan_params(
            n,
            guess_params=params,
            fit_func=fit_function,
            fit_params=fit_para,
            fit_covariance=fit_cov,
        )


def show_fit_params(
    exp: Experiment, *file_numbers, window_width=50, cov=False
):  # noqa
    file_numbers = exp.check_file_numbers(file_numbers)
    for n in file_numbers:
        params = exp.get_scan(n).fit_params
        fit_cov = exp.get_scan(n).fit_covariance
        fit_stddev = np.sqrt(np.diag(fit_cov))

        fit_values = list(zip(params, fit_stddev))
        if 0.01 < fit_values[0][0] < 1000:
            offset_string = "| Offset : {:^ 8.8f}"
        else:
            offset_string = "| Offset : {:^ 8.3e}"
        if 0.01 < fit_values[0][1] < 1000:
            offset_string += " ± {:^ 8.8f} |"
        else:
            offset_string += " ± {:^ 8.3e} |"

        number_formats = []
        for pair in fit_values[1:]:
            if 0.01 < np.abs(pair[0]) < 10000:
                number_formats.append("|{:^ 12.5f}±")
            else:
                number_formats.append("|{:^ 12.3e}±")
            if 0.01 < pair[1] < 1000:
                number_formats[-1] += "{:^ 12.6f}"
            else:
                number_formats[-1] += "{:^ 12.3e}"
            if len(number_formats) % 4 == 0:
                number_formats[-1] += "|\n"

        print(offset_string.format(*fit_values[0]))
        print(
            f'|{"Absorption Ampl":-^25}|{"Dispersion Ampl":-^25}'
            f'|{"Linewidth":-^25}|{"Position":-^25}|'
        )
        print(
            "".join(number_formats).format(
                *[i for h in fit_values[1:] for i in h]
            )
        )  # noqa

        if cov:
            print(
                "\n".join(
                    [
                        "".join(["{:10.2e}".format(item) for item in row])
                        for row in fit_cov
                    ]
                )
            )
            print()


def fit_and_plot_fmr(
    exp: Experiment,
    file_number,
    params=None,
    show_params=True,
    offset=None,
    auto=2,
    x_column=None,
    y_column=None,
    derivative=True,
    bounds=(-np.inf, np.inf),
    xbaseline=(None, None),
    cut_scale=None,
    smoothing=None,
    absolute_sigma=False,
    xfit_range=(None, None),
):
    """Fits curves to symmetric and antisymmetric components of a lorentzian
    [derivative] signal. Detects and fits automatically to any number of
    lorentzians by using the 'auto' kwarg. Set auto=# of lorentzians you expect.
    i.e. auto=2 will fit to a maximum of 2 lorentzians. For manually fitting,
    set auto=False (the default) and specify '*params' in the form [p1, p2, p3,
    p4], [p5, p6, p7, p8], ... where inital guess parameters for each lorentzian
    are organized as [absorption_amplitude, dispersion_amplitude, linewidth,
    position]. Default behavior is to fit to lorentzian derivates; set
    derivatives=False to fit to absorption lineshapes.

    For auto fitting:
    Without any extra kwargs, the function will attempt to automatically
    generate a spline and ignore background data. If fits are failing,
    modify the following *kwargs* accordingly.
    *xbaseline* (x1, x2) sets the x-values that contain your constant
    background.
    *cut_scale* (1) sets the height of the area that is ignored by fitting.
    With sensible xbaseline,
    *smoothing* (1) sets how much noise in your signal is rejected when
    generating a spline. If guesses are sharp, increase smoothing, if
    fitting results in no fits, decrease smoothing. Change by 10x.
    *xfit_range* (x1, x2) sets the x-values that contain the lorentzian(s) you
    want to fit.
    """  # noqa
    try:
        fit_fmr_exp(
            exp,
            file_number,
            params=params,
            offset=offset,
            auto=auto,
            xbaseline=xbaseline,
            cut_scale=cut_scale,
            smoothing=smoothing,
            derivative=derivative,
            x_column=x_column,
            y_column=y_column,
            xfit_range=xfit_range,
        )
    except Exception as e:
        print(f"Fit failed.\n{e}")
    try:
        if show_params:
            show_fit_params(exp, file_number, cov=False)
    except Exception as e:
        print(f"Failed to display fit.\n{e}")

    plot_guess_and_fit(
        exp,
        file_number,
        xbaseline=xbaseline,
        cut_scale=cut_scale,
        smoothing=smoothing,
        derivative=derivative,
        xfit_range=xfit_range,
        auto=auto,
    )
