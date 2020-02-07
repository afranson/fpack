"""fitting_core contains functions to automatically fit lorentzians
and provide easy ways to visualize the fitting process.

"""

from .experiment import Experiment
from .lorentz_functions import (
    absorption_dispersion_mixed,
    absorption_dispersion_derivative_mixed,
)
from .plotting_core import plot_guess_and_fit, _xbaseline_to_ibaseline
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
from scipy.interpolate import UnivariateSpline
import concurrent


def _get_auto_params(
    x, y, derivative=True, cut_scale=5, xbaseline=(None, None), smoothing=5
):
    ibaseline = _xbaseline_to_ibaseline(x, xbaseline)
    y = y - np.average(y[ibaseline[0] : ibaseline[1]])
    cut = cut_scale * np.std(y[ibaseline[0] : ibaseline[1]])
    y_cut_data = y
    y_cut_data[(y <= cut) & (y >= -cut)] = 0
    cut_data = np.vstack((x, y_cut_data))
    if not derivative:
        integral = UnivariateSpline(cut_data[0], cut_data[1], k=5)
        integral.set_smoothing_factor(smoothing)
        deriv = integral.derivative()
    else:
        deriv = UnivariateSpline(cut_data[0], cut_data[1], k=4)
        deriv.set_smoothing_factor(smoothing)
    deriv2 = deriv.derivative()
    x_roots = deriv2.roots()
    try:
        derivs = [deriv2.derivatives(x_root)[1] for x_root in x_roots]
    except IndexError as e:
        print(f"Failed to find the 1st element of the derivative of the spline at "
              f"x_roots. x_roots value: {x_roots}. Try decreasing cut_scale or "
              f"smoothing.")
        print(repr(e))
    x_d = np.dstack((x_roots, derivs))[0][(derivs[0] > 0) :]
    lorentzian_guesses = [
        (y1 - y0, (deriv(x0), 0, x1 - x0, (x1 + x0) / 2))
        for (x0, y0, x1, y1) in zip(*[np.nditer(x_d)] * 4)
    ]
    return_guesses = [
        guess for (mag, guess) in sorted(lorentzian_guesses, reverse=True)
    ]
    return return_guesses


def fit_fmr_absdisp(
    exp: Experiment,
    file_number,
    *params,
    offset=None,
    auto=False,
    x_column=None,
    y_column=None,
    derivative=True,
    bounds=(-np.inf, np.inf),
    xbaseline=(None, None),
    cut_scale=5,
    smoothing=5,
    absolute_sigma=False,
    xfit_range=(None, None),
    **fit_kwargs,
):
    """Fits curves to symmetric and antisymmetric components of a
    lorentzian [derivative] signal.  *params should be [1, 2, 3, 4],
    [5, 6, 7, 8], ...

    auto usage: set auto=# of lorentzians you expect. i.e. auto=2 will
    fit curve to a maximum of 2 lorentzians.

    """
    x_column, y_column = exp.check_xy_columns(x_column, y_column)
    x_data, y_data = exp.get_xy_data(
        file_number, x_column=x_column, y_column=y_column
    )
    num_params = len(params)
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
                    absorption_dispersion_mixed(x, *params1[(4 * n) : 4 * (n + 1)])
                    for n in range(num_params)
                ]
            )

    if xbaseline != (None, None):
        ibaseline = _xbaseline_to_ibaseline(x_data, xbaseline)
        stddev = np.std(y_data[ibaseline[0] : ibaseline[1]])
        err_list = stddev * np.ones(len(y_data[ifit_range[0] : ifit_range[1]]))
        absolute_sigma = True
    else:
        err_list = None

    params = [item for sublist in params for item in sublist]
    params.insert(0, offset)

    curve_fit_kwargs = {
        "p0": params,
        "bounds": bounds,
        "sigma": err_list,
        "absolute_sigma": absolute_sigma,
    }
    curve_fit_kwargs.update(fit_kwargs)

    x_data_tofit = x_data[ifit_range[0] : ifit_range[1]]
    y_data_tofit = y_data[ifit_range[0] : ifit_range[1]]
    try:
        fit_para, fit_cov = curve_fit(
            fit_function, x_data_tofit, y_data_tofit, **curve_fit_kwargs
        )
    except RuntimeError as e:
        print(f"Failed to fit file number: {file_number}")
        print(repr(e))
        fit_para = [0, ]
        fit_cov = [0, ]

    exp.set_scan_params(
        file_number,
        guess_params=params,
        fit_func=fit_function,
        fit_params=fit_para,
        fit_covariance=fit_cov,
    )


def fit_fmr_several(
    exp: Experiment,
    *file_numbers,
    method=0,
    widths=[0.1, 0.2, 0.3, 0.4],
    min_snr=2,
    auto=False,
    xbaseline=(None, None),
    cut_scale=10,
    smoothing=5,
    derivative=True,
    xfit_range=(None, None),
    mt=False,
):
    """
    method = 1 => auto_fitting using fit_fmr_absdisp
    method = 2 => cwt to find peaks
    method = 3 => integrate and give peak (highest) value
    """
    file_numbers = exp.check_file_numbers(file_numbers)

    fit_params = []
    fit_covs = []
    if method == 0 and mt:
        executor = concurrent.futures.ProcessPoolExecutor(2)
        futures = [
            executor.submit(
                fit_fmr_absdisp(
                    exp,
                    n,
                    auto=auto,
                    xbaseline=xbaseline,
                    cut_scale=cut_scale,
                    smoothing=smoothing,
                    derivative=derivative,
                    xfit_range=xfit_range,
                )
            )
            for n in file_numbers
        ]
        concurrent.futures.wait(futures)
    for n in file_numbers:
        data_to_fit = exp.get_xy_data(n)
        if method == 0 and not mt:
            fit_fmr_absdisp(
                exp,
                n,
                auto=auto,
                xbaseline=xbaseline,
                cut_scale=cut_scale,
                smoothing=smoothing,
                derivative=derivative,
                xfit_range=xfit_range,
            )
        elif method == 1:
            data_to_fit = exp.get_xy_data(n)
            ibaseline = _xbaseline_to_ibaseline(data_to_fit[0], xbaseline)
            integrated_data_to_fit = (
                data_to_fit[0],
                np.cumsum(
                    data_to_fit[1]
                    - np.average(data_to_fit[1][ibaseline[0] : ibaseline[1]])
                ),
            )
            index_peaks = find_peaks_cwt(
                integrated_data_to_fit[1], widths, min_snr=min_snr
            )
            field_peaks = np.array([data_to_fit[0][index] for index in index_peaks])
            fit_params.append(field_peaks)
            fit_covs.append(0)
        elif method == 2:
            data_to_fit = exp.get_xy_data(n)
            ibaseline = _xbaseline_to_ibaseline(data_to_fit[0], xbaseline)
            integrated_data_to_fit = np.cumsum(
                data_to_fit[1] - np.average(data_to_fit[1][ibaseline[0] : ibaseline[1]])
            )
            field_pk = data_to_fit[0][integrated_data_to_fit.argsort()][-1]
            fit_params.append(field_pk)
            fit_covs.append(0)
        elif mt:
            pass
        else:
            raise ValueError(
                "method value not recognized, "
                "use 0 or 1 or 2. method entered : {}".format(
                    method
                )
            )
    fit_params = np.array(fit_params)
    fit_covs = np.array(fit_covs)
    return fit_params, fit_covs


def show_fit_params(exp: Experiment, *file_numbers, window_width=50, cov=False):
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
        print("".join(number_formats).format(*[i for h in fit_values[1:] for i in h]))

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
    *params,
    show_params=True,
    offset=None,
    auto=False,
    x_column=None,
    y_column=None,
    derivative=True,
    bounds=(-np.inf, np.inf),
    xbaseline=(None, None),
    cut_scale=10,
    smoothing=5,
    absolute_sigma=False,
    xfit_range=(None, None),
):
    fit_fmr_absdisp(
        exp,
        file_number,
        *params,
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

    if show_params:
        show_fit_params(exp, file_number, cov=False)

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
