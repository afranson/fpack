"""
Andrew Franson

Package for facile extraction of routine data from the EJH Lab.

General workflow:
import fpack as fp

# Initializing Experiments and loading in/manipulating data
exp_1 = fp.EPR_Experiment()
exp_1.add_and_read_files(directory, ".txt")
exp_1_normalized = exp_1.normalize_frequency(9.73e9)
exp_2 = fp.BFMR_Experiment()
exp_2.add_and_read_files(directory, ".txt")
exp_2_rotated = exp_2.move_fileset_to_end(9)

# Plotting data from Experiments and easily creating publication quality plots
fp.plot_several(exp_2_rotated)
fp.plot_heatmap(exp_1_normalized)
fp.modify_plot(x_label="this", grid=True,
               tick_positions=(True, True, False, False),
               tick_labels=(True, False, False, False),
               ticks_in=(True, True, False, False),
               ticksize=7)

# Extracting data from Experiments to use on your own
fp.extract_data(exp_2_rotated, x_column=1, y_column=[4,5,6])

# Fitting data from Experiments
fit_exp_1 = fp.Experiment_Fit(exp_1)
fp.fit_fmr_absdisp(exp_1, fit_exp_1, 0, auto=3, cut_mult=10)
fp.fit_fmr_absdisp(exp_1, fit_exp_1, 1, auto=3, cut_mult=10)
fp.fit_fmr_several(exp_1, fit_exp_1, range(2, len(exp_1)), auto=2, cut_mult=5)
fp.show_fits(fit_exp_1)
fp.plot_fits(exp_1, fit_exp_1, waterfall=10)
"""

# How __init__.py files work

# They allow you to do import fpack as fp and access all the contents
# of fpack without having to do individual file imports, so it
# packages a bunch of modules (any .py file) into a package.

# Basic language of the file

# __all__ = ["this", "that"] specifies what gets imported from a
# statement like "from fpack import *"

# "from .function_tools.rfft_smoothing import *" imports everything in
# __all__ in the file ./function_tools/rfft_smoothing.py. If __all__
# is not defined it imports everything that doesn't start with an
# underscore (_). Things in rfft_smoothing.py will then be accessible
# via "import fpack fpack.function_from_rfft()".

__all__ = ['experiment', 'plotting_core', 'lorentz_functions']

from .experiment import *
from .lorentz_functions import *
from .plotting_core import *
from .fitting_core import *

# from fpack.core import *
# from fpack.function_tools import *
# from fpack.plotting_tools import *
# from fpack.figure_tools import *
# from fpack.fitting_tools import *
