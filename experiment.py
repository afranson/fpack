"""
Package for facile extraction of routine data from the EJH Lab.

General workflow:
import fpack as fp

exp_1 = fp.EPR_Experiment(filenames)
exp_1_normalized = exp_1.normalize_frequency(9.73e9)
exp_2 = fp.BFMR_Experiment(filenames)
exp_2_rotated = exp_2.move_fileset_to_end(9)

fp.plot_scans(exp_2_rotated)
fp.plot_heatmap(exp_1_normalized)
fp.plot_tailor(x_label="this", grid=True, figsize=(15,5),
               ticks=(True, True, False, False),
               tick_labels=(True, False, False, False),
               tick_sides=(in, in, out, both), minorticks,
               tick_label_fontsize=7)

fp.extract_data(exp_2_rotated, x_column=1, y_column=[4,5,6])
"""


import os
import re
import numpy as np
import pandas as pd
# import dill

# Color definitions for terminal output
BLUE = "\033[94m"
NORMAL = "\033[0m"

# Do you want the main functions to help you?
show_help = True

# Helper Functions #

if show_help:
    print(
<<<<<<< HEAD
        "fpack version 0.40\n"
        "Remove help via 'fp.show_help=False'\n"
        "Most of the fpack module's functionality can be found from "
        "'dir(fp)' and 'dir(fp.Experiment)'. If you are using IPython "
        "or Jupyer, then further function descriptions can be found via "
        "'fp.function?' and 'fp.function??' (or 'fp.Experiment.function?')."
        " Finally, source code can be found at 'print(fp.__file__)'\n\n"
        "Begin by creating an Experiment.\n"
        "'exp = fp.Experiment()', there are other specific types as well."
        )
=======
        '''
        fpack version 0.40
        Remove help via 'fp.show_help=False'
        Most of the fpack module's functionality can be found from \
        'dir(fp)' and 'dir(fp.Experiment)'.
        If you are using IPython or Jupyer, then further function descriptions \
        can be found via 'fp.function?' and 'fp.function??' (or 'fp.Experiment.function?').
        Finally, source code can be found at 'print(fp.__file__)'

        Begin by creating an Experiment.
        'exp = fp.Experiment()', there are other, more specific types as well.
        '''
    )
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444


def show_files(base_directory, *regexs, show_all=False):
    """Shows all files in 'base_directory' that match 'regexs'. Regular expression
    matching is done via the re package. Mulitple regexs are meant for regex
    beginners so they can just match strings like "VoltageSweep" and "300K" to
    match all scans that contain both those exact strings and no others. More
    advanced users can use one regex to match for any desired files. Entering no
    regex matches all files in that directory.

    """  # noqa
    files = os.listdir(base_directory)

    if len(regexs) == 0:
        patterns = [re.compile("")]
    else:
        try:
            patterns = [re.compile(regex) for regex in regexs]
        except TypeError:
            raise TypeError(
                f"All 'regexs' must be strings. You have "
                f"currently entered regexs={regexs}."
            )

    printable_files = []
    for n, f in enumerate(files):
        if all(bool(patt.search(f)) for patt in patterns):
            printable_files.append(f"{BLUE}{n:4d} : {f}{NORMAL}")
        elif show_all:
            printable_files.append(f"{n:4d} : {f}")

    print("Files matching pattern:")
    print("\n".join(printable_files))
    print()


# ------- Scan Class ------- #


class Scan:
    """Holds all of a file's typical information in one nice neat package.

    """

    def __init__(
        self,
        *,
        filename=None,
        info=None,
        axes=None,
        data=None,
        guess_params=None,
        fit_func=None,
        fit_params=None,
        fit_covariance=None,
    ):
        self.filename = filename
        self.info = info
        self.axes = axes
        self.data = data
        self.guess_params = guess_params
        self.fit_func = fit_func
        self.fit_params = fit_params
        self.fit_covariance = fit_covariance

    def set_scan_params(
        self,
        *,
        filename=None,
        info=None,
        axes=None,
        data=None,
        guess_params=None,
        fit_func=None,
        fit_params=None,
        fit_covariance=None,
    ):
        if not (filename is None):
            self.filename = filename
        if not (info is None):
            self.info = info
        if not (axes is None):
            self.axes = axes
        if not (data is None):
            self.data = data
        if not (guess_params is None):
            self.guess_params = guess_params
        if not (fit_func is None):
            self.fit_func = fit_func
        if not (fit_params is None):
            self.fit_params = fit_params
        if not (fit_covariance is None):
            self.fit_covariance = fit_covariance

    def copy_scan(self):
        return Scan(
            filename=self.filename,
            info=np.copy(self.info),
            axes=np.copy(self.axes),
            data=np.copy(self.data),
            guess_params=np.copy(self.guess_params),
            fit_func=self.fit_func,
            fit_params=np.copy(self.fit_params),
            fit_covariance=np.copy(self.fit_covariance),
        )


# ------- Experiment Class ------- #


class Experiment:
    """Base class to inheret from to create more specific measurement classes.
    Easily load data by setting the defaults to sensible values.

    """

    def __init__(
        self,
        default_x_column=0,
        default_y_column=1,
        default_descriptive_rows=0,
        default_axis_label_row=0,
        default_data_start_row=1,
        default_delim_whitespace=True,
        default_sep=",",
    ):
        self.default_x_column = default_x_column
        self.default_y_column = default_y_column
        self.default_descriptive_rows = default_descriptive_rows
        self.default_axis_label_row = default_axis_label_row
        self.default_data_start_row = default_data_start_row
        self.default_delim_whitespace = default_delim_whitespace
        self.default_sep = default_sep
        self.scans = list()
        self.storage = dict()
        if show_help:
            print(
<<<<<<< HEAD
                "Now add files to the experiment."
                "exp.load_files(r\"~/path/to/dir\" (or C:\\path\\to\\dir), "
                "r\"[regex]*.[to\\d]?(_|-)[match]+\")"
            )

=======
                '''
                Now add files to the experiment.
                exp.load_files(r"path/to/dir", r"[regex]*.[to\d]?(_|-)[match]+")
                '''
            )
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
    def __len__(self):
        return len(self.scans)

    def __iter__(self):
        return iter(self.scans)

    def __repr__(self):
        try:
            name = [k for k, v in globals().items() if v is self][0]
            return f"Experiment '{name:s}'"
        except IndexError:
            return f"Experiment (failed to find name globals)"

    # ------- Scan Manipulation Functions ------- #

    def get_scan(self, file_number):
        return self.scans[file_number]

    def add_scan(
        self,
        position=None,
        *,
        filename=None,
        info=None,
        axes=None,
        data=None,
        guess_params=None,
        fit_func=None,
        fit_params=None,
        fit_covariance=None,
        scan=None,
    ):
        if scan is None:
            scan = Scan(
                filename=filename,
                info=info,
                axes=axes,
                data=data,
                guess_params=guess_params,
                fit_func=fit_func,
                fit_params=fit_params,
                fit_covariance=fit_covariance,
            )
        else:
            scan = scan.copy_scan()
        if position:
            self.scans.insert(position, scan)
        else:
            self.scans.append(scan)

    def remove_scans(self, *file_numbers):
        for n in reversed(file_numbers):
            del self.scans[n]

    def set_scan_params(
        self,
        file_number,
        *,
        filename=None,
        info=None,
        axes=None,
        data=None,
        guess_params=None,
        fit_func=None,
        fit_params=None,
        fit_covariance=None,
    ):
        scan = self.scans[file_number]
        scan.set_scan_params(
            filename=filename,
            info=info,
            axes=axes,
            data=data,
            guess_params=guess_params,
            fit_func=fit_func,
            fit_params=fit_params,
            fit_covariance=fit_covariance,
        )

    # ------- Data Manipulation Functions ------- #

    def add_files(self, base_directory, *regexs):
        """Adds all files in 'base_directory' that match 'regexs'. Regular expression
        matching is done via the re package. Mulitple regexs are meant for regex
        beginners so they can just match strings like "VoltageSweep" and "300K"
        to match all scans that contain both those exact strings and no others.
        More advanced users can use one regex to match for any desired files.
        Entering no regex matches all files in that directory.

        """  # noqa
        files = os.listdir(base_directory)

        if len(regexs) == 0:
            patterns = [re.compile("")]
        else:
            try:
                patterns = [re.compile(regex) for regex in regexs]
            except TypeError:
                raise TypeError(
                    f"All 'regexs' must be strings. You have "
                    f"currently entered regexs={regexs}."
                )
        [
            self.add_scan(filename=base_directory + os.sep + f)
            for f in files
            if all(bool(patt.search(f)) for patt in patterns)
        ]

    def read_files(
        self,
        *file_numbers,
        descriptive_rows=None,
        axis_label_row=None,
        data_start_row=None,
        sep=None,
        delim_whitespace=None,
        transpose=True,
        **read_csv_kwargs,
    ):
        """Reads all provided file numbers (enter nothing to read all of them)

        """
        file_numbers = self.check_file_numbers(file_numbers)
        if descriptive_rows is None:
            descriptive_rows = self.default_descriptive_rows
        if axis_label_row is None:
            axis_label_row = self.default_axis_label_row
        if data_start_row is None:
            data_start_row = self.default_data_start_row
        if sep is None:
            sep = self.default_sep
        if delim_whitespace is None:
            delim_whitespace = self.default_delim_whitespace
        for file_number in file_numbers:
            filename = self.get_scan(file_number).filename
            temp_header = []
            with open(filename, "r") as f:
                temp_header = [f.readline() for _ in range(descriptive_rows + 1)]  # noqa
                self.set_scan_params(file_number, info=temp_header)
                _ = [f.readline() for _ in range(axis_label_row - descriptive_rows - 1)]  # noqa
                axes_line = f.readline()
                if delim_whitespace:
                    axes = axes_line[:-1].split()
                else:
                    axes = axes_line[:-1].split(sep)
                self.set_scan_params(file_number, axes=axes)
                read_csv_kwargs = {
                    "skiprows": data_start_row - axis_label_row - 1,
                    "sep": sep,
                    "delim_whitespace": delim_whitespace,
                    "header": None,
                    **read_csv_kwargs,
                }
                try:
                    data = np.array(pd.read_csv(f, **read_csv_kwargs))
                except pd.errors.ParserError as e:
                    raise TypeError(f"pandas could not read file: {f}.\n{repr(e)}")  # noqa
            if transpose:
                self.set_scan_params(file_number, data=data.T)
            else:
                self.set_scan_params(file_number, data=data)

    def load_files(
        self,
        base_directory,
        *regexs,
        show=True,
        dill_file=None,
        descriptive_rows=None,
        axis_label_row=None,
        data_start_row=None,
        sep=None,
        delim_whitespace=None,
        **read_csv_kwargs,
    ):
        """Adds and reads files into the Experiment using the add_files and the
        read_files methods. Also allows for loading of a previous saved
        Experiment via 'dill_file'. It will try to load 'dill_file' first and
        then default to adding and reading files if that fails. Also accepts all
        pandas.read_csv key work arguments for highly customized file reading.

        """  # noqa
        try:
            # self.load_exp(dill_file)
            raise FileNotFoundError
            return None
        except FileNotFoundError:
            print(
                f"Failed to load in dill_file, {dill_file},"
                f"proceeding to load files regularly."
            )
        except TypeError:  # Move on if dill_file is None
            pass
        self.add_files(base_directory, *regexs)
        self.read_files(
            descriptive_rows=descriptive_rows,
            axis_label_row=axis_label_row,
            data_start_row=data_start_row,
            delim_whitespace=delim_whitespace,
            sep=sep,
            **read_csv_kwargs,
        )
        if show:
            self.show_files()
        if show_help:
            print(
<<<<<<< HEAD
                "Now you can examine, plot, "
                "and extract the loaded data easily.\n"
                "The various exp.show_xxx() functions "
                "reveal info about loaded files.\n"
                "fp.plot_scans(exp) and the other "
                "fp.plot_xxx(exp) functions to plot.\n"
                "fp.plot_package_help() for more info. \n"
                "exp.get_xy_data() and other exp.get_xxx() "
                "to retrieve information from the Experiment.\n"
                "fp.fit_fmr_absdisp() and fp.fit_fmr_several()"
                " will fit one or more scans manually or automatically.\n"
                "fp.plot_fits() will plot fit(s) along with the data."
=======
                '''
                Now you can examine the loaded data, plot it, and extract it easily.
                The various exp.show_xxx() functions reveal info about loaded files.
                fp.plot_scans(exp) and the other fp.plot_xxx(exp) functions to plot.
                fp.plot_tailor() to modify plots (line types, fonts, legends, etc.)
                exp.get_xy_data() and other exp.get_xxx() to retrieve information
                from the Experiment.
                '''
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
            )

    def set_axes(self, *axis_labels):
        """Overrides the axes of the Experiment in the event of poor default
        reading.

        """
        scan: Scan
        for scan in self:
            scan.set_scan_params(axes=axis_labels)

    def show_file_head(self, file_number, num_rows=20):
        """Displays the first 'num_rows' of a file that has been added to the
        Experiment along with row numbers to assist in setting up
        reading parameters.

        """
        print("File Preview")
        print("Line: Contents")
        file_to_read = self.get_scan(file_number).filename
        with open(file_to_read) as f:
            for n in range(num_rows):
                print("{:4d}".format(n) + " : " + f.readline()[:-1])
        print()

    def show_files(self):
        """Displays all files added to the Experiment.

        """
        print("Files loaded into data structure:")
        print(
            "\n".join(
                [
                    "{:4d}".format(n) + " : " + scan.filename
                    for n, scan in enumerate(self)
                ]
            )
        )
        print()

    def show_files_info(self, *file_numbers, rows=None, view=True):
        """Shows information stored at the beginning of a file.

        """
        file_numbers = self.check_file_numbers(file_numbers)
        if rows is None:
            rows = list(range(len(self.get_scan(file_numbers[0]).info)))
        rows = [rows[i: i + 3] for i in range(0, len(rows), 3)]
        rows[-1] = (rows[-1] + [None] * 3)[:3]

        print("|" + "-" * (25 + 19 * len(rows[0])) + "|")
        row_strings = []
        row0_string = "|{:^25}|".format(r"File Name\Row")
        row_strings = [row0_string] + [f"|{'':^25}|"] * (len(rows) - 1)
        for n, superrow in enumerate(rows):
            for row in superrow:
                if row is None:
                    row_strings[n] += f"{'':<18.18}|"
                else:
                    row_info = self.get_scan(0).info[row]
                    row_strings[n] += f"{row_info.split(':')[0]:^18.16}|"
        for n, _ in enumerate(rows):
            print(row_strings[n])
        print("|" + "-" * (25 + 19 * len(rows[0])) + "|")

        for file_num in file_numbers:
            scan = self.get_scan(file_num)
            filename = re.search(r"[\\/][\w\.\s]+$", scan.filename).group()[1:]
            filenames = []
            for n in range(len(rows)):
                if n == 0:
                    filenames.append(filename[:18])
                else:
                    filenames.append(filename[18 + 23 * (n - 1): 18 + 23 * n])
            file_strings = []
            file_info = f"|{file_num:4d}: {filenames[0]:^19.18}|"
            file_strings = [file_info] + [""] * (len(rows) - 1)
            for n, superrow in enumerate(rows):
                for row in superrow:
                    if n > 0 and file_strings[n] == "":
                        file_strings[n] = f"|{filenames[n]:^25.23}|"
                    if row is None:
                        file_strings[n] += f"{'':<18.18}|"
                        pass
                    else:
                        row_info = self.get_scan(file_num).info[row]
                        file_strings[n] += f"{row_info.split(':')[1][:-1]:<18.16}|"  # noqa
            for n, _ in enumerate(rows):
                print(file_strings[n])
            print("|" + "-" * (25 + 19 * len(rows[0])) + "|")

    def show_file_data(self, file_number, num_rows=21):
        """Prints the first 'num_rows' of data loaded into the Experiment for
        the specified 'file_number'.

        """
        print("Data Preview:\n")
        print(
            " ".join([f"{element:<16}" for element in self.get_scan(file_number).axes])  # noqa
        )
        for data_row in self.get_scan(file_number).data.T[0:num_rows]:
            try:
                print(" ".join([f"{element:<16.10}" for element in data_row]))
            except ValueError:
                print(" ".join([f"{element:<16}" for element in data_row]))
        print()

    def check_file_numbers(self, file_numbers):
        """Replaces and empty tuple of file numbers with the complete tuple
        allowing for easy iteration over all file numbers in an Experiment.

        """
        if file_numbers == ():
            file_numbers = list(range(len(self)))
        return file_numbers

    def check_xy_columns(self, x_column, y_column):
        """Checks if x_column or y_column is None. If one is, it returns the
        Experiment's default for the appropriate column.

        """
        if x_column is None:
            x_column = self.default_x_column
        if y_column is None:
            y_column = self.default_y_column
        return x_column, y_column

    def get_xy_data(self, *file_numbers, x_column=None, y_column=None):
        """Returns the Experiment's x and y data for the specified column
        numbers and files.

        """
        file_numbers = self.check_file_numbers(file_numbers)
        x_column, y_column = self.check_xy_columns(x_column, y_column)

        return_array = []
        for file_number in file_numbers:
            data_to_extract = self.get_scan(file_number).data
            x_data = data_to_extract[x_column]
            y_data = data_to_extract[y_column]
            return_array.append([x_data, y_data])

        if len(file_numbers) == 1:
            return_array = return_array[0]

        return return_array

    def _get_fit_params(self, *file_numbers, fit_param_indexes):
        try:
            values = np.zeros((len(file_numbers), len(fit_param_indexes)))
        except TypeError:
            values = np.zeros(len(file_numbers))
        for n, file_num in enumerate(file_numbers):
            scan = self.get_scan(file_num)
            fit_params = scan.fit_params
            try:
                values[n] = fit_params[fit_param_indexes]
            except TypeError:
                for m, i in enumerate(fit_param_indexes):
                    try:
                        values[n, m] = fit_params[i]
                    except IndexError:
                        values[n, m] = np.nan
        if len(file_numbers) == 1:
            return values[0]
        return values

    def _get_file_and_info_item(self, *file_numbers, regex, repl, match_number):  # noqa
        pattern = re.compile(regex)
        values = np.zeros(len(file_numbers), dtype='<U100')
        for n, file_num in enumerate(file_numbers):
            scan = self.get_scan(file_num)
            metadata = np.hstack((scan.filename, scan.info))
            temp_values = []
            for md in metadata:
                temp_value = pattern.sub(repl, md)
                if temp_value != md:
                    temp_values.append(temp_value)
            try:
                values[n] = temp_values[match_number]
            except IndexError:
                values[n] = np.nan
        try:
            values = np.asarray(values, dtype=np.float)
        except ValueError:
            pass
        if len(file_numbers) == 1:
            return values[0]
        else:
            return values

    def get_metadata(
        self,
        *file_numbers,
        regex=r".*?(-?\d+(?:,\d+)*(?:e\d+)?(?:\.\d+(?:e\d+)?)?).*$",
        repl=r"\1",
        match_number=0,
        fit_param_indexes=None,
        return_file_numbers=None,
    ):
        """Extracts metadata from the filename and info of scans and prints it out in
        the format specified by 'repl' using re.sub(regex, repl, metadata). See
        re.sub() documentation for more details. Also can extract one or several
        fit_params by index.

        Quick, basic usage: filename="filename_freq_9.445GHz.txt". To extract
        the "9.445", the default will work fine, it extracts all numbers from
        the metadata, you just have to access the correct number via
        match_number. Another way would be regex=r".*?freq_([\d\.]+).*" with
        repl=r"\1".

        You have to match everything, hence the ".*?" at the beginning and the
        ".*" at the end. "freq_" gives the regex a position in the string to
        start extracting from. Parentheses capture the pattern inside them -
        each set of parentheses can be recalled using the appropriate r"\#"
        starting with 1 for the 1st set (i.e. \1). "[\d\.]+" will match all
        numbers and periods found after the "freq_" string, in this case,
        "9.445". Since it is in parenthesis, it will be capture group 1 and can
        be recalled by repl=r"\1". To get "9.445 GHz" as the result, this can be
        extended to regex=r".*?freq_([\d\.]+)(GHz).*" with repl=r"\1 \2" or done
        manually via repl=r"\1 GHz".

        If multiple metadata fields match (filename and an info field), then
        'match_number' breaks the tie and lets you choose which one you want.
        This can be avoided by careful selection of your regex, however.

<<<<<<< HEAD
        """  # noqa
=======
        """
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
        file_numbers = self.check_file_numbers(file_numbers)
        if not (fit_param_indexes is None):
            return self._get_fit_params(*file_numbers, fit_param_indexes=fit_param_indexes)  # noqa

        if not (return_file_numbers is None):
            return list(file_numbers)

        return self._get_file_and_info_item(
            *file_numbers, regex=regex, repl=repl, match_number=match_number
        )

    # ------- Experiment Data Modification Functions ------- #

    # ------- Experiment Management Functions ------- #

    def subsection(self, *file_numbers, exclude=None):
        file_numbers = self.check_file_numbers(file_numbers)
        if exclude is None:
            exclude = []

        temp_file_numbers = list(file_numbers)
        temp_experiment = Experiment()
        for number in reversed(temp_file_numbers):
            if number in exclude:
                del temp_file_numbers[number]
        for n in temp_file_numbers:
            temp_experiment.add_scan(scan=self.get_scan(n))
        return temp_experiment

    def move_scan_to_start(self, file_number):
        self.scans = self.scans[file_number:] + self.scans[:file_number]

    def move_scan_to_end(self, file_number):
        self.move_scan_to_start(file_number + 1)

    # def save_exp(self, filename=None):
    #     if filename is None:
    #         filename = str(len(self))
    #     filename += ".dill"
    #     with open(filename, "wb") as f:
    #         dill.dump(self, f)
    #     print(f"Saved experiment to {filename}.")

    # def load_exp(self, filename):
    #     with open(filename, "rb") as f:
    #         temp_exp = dill.load(f)
    #         for scan in temp_exp:
    #             self.add_scan(scan=scan)

    # ------- Fitting Manipulation ------- #

    def get_xy_fit(self, file_number, x_fit=None, x_column=None, x_density=1):
        if x_fit is None:
            x, _ = self.get_xy_data(file_number, x_column=x_column)
            x_fit = np.linspace(x[0], x[-1], x_density * len(x))
        func = self.get_scan(file_number).fit_func
        params = self.get_scan(file_number).fit_params
        return x_fit, func(x_fit, *params)

    def get_xy_guess(self, file_number, x_fit=None, x_column=None, x_density=1):  # noqa
        if x_fit is None:
            x, _ = self.get_xy_data(file_number, x_column=x_column)
            x_fit = np.linspace(x[0], x[-1], x_density * len(x))
        func = self.get_scan(file_number).fit_func
        params = self.get_scan(file_number).guess_params
        return x_fit, func(x_fit, *params)

    def get_xy_werror(self, file_number, x_fit=None, x_column=None, x_density=1):  # noqa
        if x_fit is None:
            x, _ = self.get_xy_data(file_number, x_column=x_column)
            x_fit = np.linspace(x[0], x[-1], x_density * len(x))
        func = self.get_scan(file_number).fit_func
        params = self.get_scan(file_number).fit_params
        cov = self.get_scan(file_number).fit_covariance
        stderr = np.sqrt(np.diag(cov))
        y_fit = func(x_fit, *params)
        y_upper = func(x_fit, *(params + stderr))
        y_lower = func(x_fit, *(params - stderr))
        return x_fit, y_fit, y_lower, y_upper

    def get_all_fit_params(self):
        return_params = np.asarray(self.get_scan(0).fit_params)
        for n, scan in enumerate(self):
            return_params = np.vstack((return_params, scan.fit_params))
        return return_params[1:]


class EPR_Experiment(Experiment):
    """Class for loading and manipulating data recieved from the Bruker
    EPR.
    """

    def __init__(self):
        super().__init__(
            default_x_column=1,
            default_y_column=2,
            default_descriptive_rows=-1,
            default_axis_label_row=0,
            default_data_start_row=2,
            default_delim_whitespace=True,
            default_sep=",",
        )

    def normalize_frequency(self, *file_numbers, overwrite=False, desired_frequency=10):  # noqa
        """Normalizes cavity EPR data to a certain frequency to account for
        frequency changes due returning of the cavity for different
        sample geometries.

        """
        file_numbers = self.check_file_numbers(file_numbers)

        if not overwrite:
            return_experiment = EPR_Experiment()
            for scan in self:
                return_experiment.add_scan(scan=scan)
        else:
            return_experiment = self

        for file_number in file_numbers:
            scan = self.get_scan(file_number)
            scan_freq = float(scan.filename.split("GHz")[0].split("_")[-1])
            scan_index = scan.data[0]
            scan_fields = scan.data[1]
            scan_intens = scan.data[2]
            new_fields = scan_fields * desired_frequency / scan_freq
            new_data = np.vstack((scan_index, new_fields, scan_intens))
            return_experiment.set_scan_params(file_number, data=new_data)
        return return_experiment

    def set_axes(self, *axis_labels):
        if axis_labels == ():
            super().set_axes("Index", "Field (G)", "Intensity (a.u.)")
        else:
            super().set_axes(*axis_labels)


class BFMR_Experiment(Experiment):
    """Class for loading and manipulating data recieved from the custom
    BFMR setup measured using the lock-in in PRB 1135.
    """

    def __init__(self):
        super().__init__(
            default_x_column=0,
            default_y_column=1,
            default_descriptive_rows=13,
            default_axis_label_row=14,
            default_data_start_row=15,
            default_delim_whitespace=False,
            default_sep="\t",
        )


class PNA_Experiment(Experiment):
    """Class for loading and manipulating data recieved from the custom
    BFMR setup measured using the PNA (VNA) in PRB 1135.
    """

    def __init__(self):
        super().__init__(
            default_x_column=0,
            default_y_column=5,
            default_descriptive_rows=11,
            default_axis_label_row=12,
            default_data_start_row=13,
            default_delim_whitespace=False,
            default_sep="\t",
        )

<<<<<<< HEAD

def loadtest():
    """
    Loads a basic experiment into 'test_exp' for testing purposes.
    """
    test_exp = Experiment()
    test_exp.add_scan(
        filename=(
            r"/path/to/your/file"
            r"/Written_8.83V_and_stuff_abo6.6 GHzdomness.ascii"
=======
def loadtest():
    test_exp = Experiment()
    test_exp.add_scan(
        filename=(
            r"C:\Users\frans\Documents\Org_Research_Notebook"
            r"\Written_8.83V_and_stuff_abo6.6 GHzdomness.ascii"
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
        ),
        info=[
            "Current (A): 5\n",
            "Voltage (V): 13\n",
            "Field (G): 3500.23434\n",
            "Sweep Direction: Up\n",
<<<<<<< HEAD
            "Temperature(K): 273\n",
=======
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
        ],
        axes=["Ax0", "Ax1", "Ax2"],
        data=np.array([[1, 2, 3], [1, 2, 3], [7, 8, 9]]),
    )
    test_exp.add_scan(
        filename=(
<<<<<<< HEAD
            r"/path/to/your/file"
            r"/Written_4.77V with 9.83 GHz and whatever.txt"
=======
            r"C:\Users\frans\Documents\Org_Research_Notebook"
            r"\Written_4.77V with 9.83 GHz and whatever.txt"
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
        ),
        info=[
            "Current (A): 5.5\n",
            "Voltage (V): 42\n",
            "Field (G): 3526.4\n",
            "Sweep Direction: Down\n",
<<<<<<< HEAD
            "Temperature(K): 77\n",
=======
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
        ],
        axes=["Ax0", "Ax1", "Ax2"],
        data=np.array([[4, 5, 6], [7, 8, 9], [12, 0, 4]]),
    )
<<<<<<< HEAD
    return test_exp
=======
>>>>>>> 92efe888ee7905aa418ef06b567d78c180610444
