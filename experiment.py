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

# TODO Add filename/ file number to fit_and_plot_fmr command - can it label itself? just change default?
# TODO Add emcee support for fitting

import os
import re
import numpy as np
import pandas as pd

# import dill

# Color definitions for terminal output
BLUE = "\033[94m"
NORMAL = "\033[0m"

# Helper Functions #


def exp_package_help():
    print(
        "fpack version 0.80\n"
        "Most of the fpack module's functionality can be found from "
        "dir(fp) and dir(fp.Experiment). If you are using IPython "
        "or Jupyer, then further function descriptions can be found via "
        "fp.function? and fp.function?? (or fp.Experiment.function?).\n"
        "Source code can be found at print(fp.__file__)\n"
        "View other help with plot_package_help() and fit_package_help()."
        "\n\n"
        "Begin by creating an Experiment.\n"
        'exp = fp.Experiment(r"~/path/to/dir" (or C:\\path\\to\\dir), '
        'r"[regex]*.[to\\d]?(_|-)[match]+")'
        "Add and load extra files with\n"
        "exp.load_files(same args as Experiment)"
        "\n\n"
        "After loading data, you can fit, plot, "
        "and extract the loaded data easily.\n"
        "The various exp.show_xxx() functions "
        "reveal info about loaded files.\n"
        "fp.plot_scans(exp) and the other "
        "fp.plot_xxx(exp) functions to plot.\n"
        "fp.fit_and_plot_fmr() fits a single file and "
        "shows behind the scenes of the automated fitting.\n"
        "fp.fit_fmr_exp()"
        " will fit one or more lorentzian scans manually or automatically.\n"
        "fp.fit_exp() will more generally fit to whatever function yo want.\n"
        "fp.plot_fits() will plot fit(s) along with the data."
        "\n\n"
        "exp.get_xy_data(), exp.get_data(), exp.get_fit_params() and other "
        "exp.get_xxx() to retrieve information from the Experiment.\n"
        "Wherever a *file_numbers appears, leaving it blank will apply "
        "all files in the experiment. Acquiring a certain range can be "
        "done via *range(start, end + 1)."
    )
exp_package_help()


def _pandas_error(exception):
    raise ValueError(
        "Encountered issues reading your files."
        "Common issues are:\n"
        "- Check that you are only reading the files you want. Do"
        " exp.show_files() to see all the files that have been "
        "added. Reload the Experiment and add more regexes or use"
        "more restraining regexes in you load_files call.\n"
        "- Check that you are not reading the headers from your"
        " file into the data. Correct this by increasing the "
        "default_data_start_row kwarg to Experiment.\n"
        "- Check that the data in your files is uniform.\n"
        f"\nPandas error below.\n{repr(exception)}"
    )


def _get_data_start(filename, *, lines_to_profile=100):
    nums_per_line = [0] * 100
    delims = [" ", "\t", ",", ";"]
    with open(filename) as f:
        i = 0
        for line in f:
            if i >= lines_to_profile:
                break
            i += 1
            active_delims = [d for d in delims if d in line]
            for delim in active_delims:
                try:
                    if delim == " ":
                        delim = None
                    [float(x) for x in line.split(delim)]
                    nums_per_line[i] = len(line.split(delim))
                except:  # noqa
                    pass
    for i in reversed(nums_per_line):
        if i:
            num_data_columns = i
            break
    guess_data_start_row = nums_per_line.index(num_data_columns) - 1
    guess_axes_label_row = guess_data_start_row - 1
    guess_descriptive_rows = guess_axes_label_row - 1
    guess_delim = active_delims[0]
    return (
        guess_descriptive_rows,
        guess_axes_label_row,
        guess_data_start_row,
        guess_delim,
    )


def show_files(base_directory, *regexs, show_all=False):
    """Shows all files in 'base_directory' that match 'regexs'. Regular expression
    matching is done via the re package. Mulitple regexs are meant for regex
    beginners so they can just match strings like "VoltageSweep" and "300K" to
    match all scans that contain both those exact strings and no others. More
    advanced users can use one regex to match for any desired files. Entering no
    regex matches all files in that directory.

    """  # noqa
    files = sorted(os.listdir(base_directory))

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
        if filename is not None:
            self.filename = filename
        if info is not None:
            self.info = info
        if axes is not None:
            self.axes = axes
        if data is not None:
            self.data = data
        if guess_params is not None:
            self.guess_params = guess_params
        if fit_func is not None:
            self.fit_func = fit_func
        if fit_params is not None:
            self.fit_params = fit_params
        if fit_covariance is not None:
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
        base_directory=None,
        *regexes,
        dill_file=None,
        show=True,
        default_x_column=0,
        default_y_column=1,
        descriptive_rows=None,
        axis_label_row=None,
        data_start_row=None,
        sep=None,
        **csv_kwargs,
    ):
        self.default_x_column = default_x_column
        self.default_y_column = default_y_column
        self.scans = list()
        self.storage = dict()
        if base_directory is not None:
            self.load_files(
                base_directory,
                *regexes,
                dill_file=dill_file,
                show=show,
                descriptive_rows=descriptive_rows,
                axis_label_row=axis_label_row,
                data_start_row=data_start_row,
                sep=sep,
                **csv_kwargs,
            )
        else:
            print(
                "Now add files to the experiment.\n"
                'exp.load_files(r"~/path/to/dir" (or C:\\path\\to\\dir), '
                'r"[regex]*.[to\\d]?(_|-)[match]+")'
            )

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
        try:
            return self.scans[file_number]
        except IndexError as e:
            print(f"file {file_number} is not in the experiment.\n{e}")

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
        for n in reversed(sorted(file_numbers)):
            filename = self.get_scan(n).filename
            print(f"Removing file {n}: {filename}")
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
        files = sorted(os.listdir(base_directory))

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
        transpose=True,
        **read_csv_kwargs,
    ):
        """Reads all provided file numbers (enter nothing to read all of them)
        """
        file_numbers = self.check_file_numbers(file_numbers)
        guess_params = _get_data_start(self.get_scan(file_numbers[0]).filename)
        inputs = (descriptive_rows, axis_label_row, data_start_row, sep)
        params = [0] * len(inputs)
        all_params = zip(guess_params, inputs)
        for n, param in enumerate(all_params):
            for option in param:
                if option is not None:
                    params[n] = option
        descriptive_rows, axis_label_row, data_start_row, sep = params
        if sep == " ":
            sep = "\s+"  # noqa
        for file_number in file_numbers:
            filename = self.get_scan(file_number).filename
            temp_header = []
            with open(filename, "r") as f:
                temp_header = [
                    f.readline() for _ in range(descriptive_rows + 1)
                ]  # noqa
                self.set_scan_params(file_number, info=temp_header)
                _ = [
                    f.readline()
                    for _ in range(axis_label_row - descriptive_rows - 1)
                ]
                axes_line = f.readline()
                if sep == "\s+":  # noqa
                    axes = axes_line[:-1].split()
                else:
                    axes = axes_line[:-1].split(sep)
                if len(axes) in [0, 1]:
                    axes = ["x", "y1", "y2", "y3", "y4", "y5", "y6", "y7"]
                self.set_scan_params(file_number, axes=axes)
                read_csv_kwargs = {
                    "skiprows": data_start_row - axis_label_row - 1,
                    "sep": sep,
                    "header": None,
                    **read_csv_kwargs,
                }
                try:
                    data = np.array(pd.read_csv(f, **read_csv_kwargs))
                except Exception as e:
                    _pandas_error(e)
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
            sep=sep,
            **read_csv_kwargs,
        )
        if show:
            self.show_files()

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
        print("Remove unwanted files with exp.remove_files(n1, n2, ...)")
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
        rows = [rows[i : i + 3] for i in range(0, len(rows), 3)]
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
                    filenames.append(filename[18 + 23 * (n - 1) : 18 + 23 * n])
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
                        file_strings[
                            n
                        ] += f"{row_info.split(':')[1][:-1]:<18.16}|"  # noqa
            for n, _ in enumerate(rows):
                print(file_strings[n])
            print("|" + "-" * (25 + 19 * len(rows[0])) + "|")

    def show_file_data(self, file_number, num_rows=21):
        """Prints the first 'num_rows' of data loaded into the Experiment for
        the specified 'file_number'.

        """
        print("Data Preview:\n")
        print(
            " ".join(
                [
                    f"{element:<16}"
                    for element in self.get_scan(file_number).axes
                ]
            )  # noqa
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
        return_array = np.zeros(
            (
                len(file_numbers),
                2,
                self.get_scan(file_numbers[0]).data.shape[1],
            )
        )
        for n, file_number in enumerate(file_numbers):
            data_to_extract = self.get_scan(file_number).data
            x_data = data_to_extract[x_column]
            y_data = data_to_extract[y_column]
            return_array[n] = [x_data, y_data]
        if len(file_numbers) == 1:
            return_array = return_array[0]
        return return_array

    def get_data(self, *file_numbers):
        """Returns the Experiment's x and y data for the specified column
        numbers and files.

        """
        file_numbers = self.check_file_numbers(file_numbers)
        return_array = np.zeros(
            (len(file_numbers), *self.get_scan(file_numbers[0]).data.shape)
        )
        for n, file_number in enumerate(file_numbers):
            return_array[n] = self.get_scan(file_number).data
        if len(file_numbers) == 1:
            return_array = return_array[0]
        return return_array

    def get_fit_params(self, *file_numbers, fit_param_indices=None, stderr=None):
        file_numbers = self.check_file_numbers(file_numbers)
        if fit_param_indices is None:
            fit_param_indices = range(
                self.get_scan(file_numbers[0]).fit_params.size
            )
        fit_param_indices = np.array(fit_param_indices)
        try:
            values = np.zeros((len(file_numbers), len(fit_param_indices)))
        except TypeError:
            values = np.zeros((len(file_numbers)))
        for n, file_num in enumerate(file_numbers):
            scan = self.get_scan(file_num)
            if stderr is not None:
                fit_params = np.sqrt(np.diag(scan.fit_covariance))
            else:
                fit_params = scan.fit_params
            if fit_params is None:  # This file has no fit data
                print(f"Warning: file {file_num} has no fit data.")
                values[n] = np.nan
                continue
            try:
                values[n] = fit_params[fit_param_indices]
            except IndexError:  # if some index doesn't exist
                print(
                    f"Warning: file {file_num} has insufficient "
                    f"fit.\nAsked for elements {fit_param_indices} of "
                    f"fit_params which is : {fit_params}\n"
                    "Only returning valid values"
                )
                reduced_indices = fit_param_indices[
                    fit_param_indices < len(fit_params)
                ]
                values[n, : len(reduced_indices)] = fit_params[reduced_indices]
            except Exception:
                reduced_size = len(fit_params)
                values[n, :reduced_size] = fit_params
        if len(file_numbers) == 1:
            return values[0]
        return values

    def _get_file_and_info_item(
        self, *file_numbers, regex, repl, match_number
    ):
        pattern = re.compile(regex)
        values = np.zeros(len(file_numbers), dtype="<U100")
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
        fit_param_indices=None,
        stderr=None,
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

        """  # noqa
        file_numbers = self.check_file_numbers(file_numbers)
        if fit_param_indices is not None:
            return self.get_fit_params(
                *file_numbers, fit_param_indices=fit_param_indices, stderr=stderr
            )

        if return_file_numbers is not None:
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

    def get_fit_func(self, file_number):
        return self.get_scan(file_number).fit_func

    def get_xy_fit(self, file_number, x_fit=None, x_column=None, x_density=1):
        if x_fit is None:
            x, _ = self.get_xy_data(file_number, x_column=x_column)
            x_fit = np.linspace(x[0], x[-1], x_density * len(x))
        func = self.get_scan(file_number).fit_func
        params = self.get_scan(file_number).fit_params
        if func is None or params is None:
            print(f"Warning: file {file_number} has no fit data.")
            return x_fit, None
        return x_fit, func(x_fit, *params)

    def get_xy_guess(
        self, file_number, x_fit=None, x_column=None, x_density=1
    ):
        if x_fit is None:
            x, _ = self.get_xy_data(file_number, x_column=x_column)
            x_fit = np.linspace(x[0], x[-1], x_density * len(x))
        func = self.get_scan(file_number).fit_func
        params = self.get_scan(file_number).guess_params
        if func is None or params is None:
            print(f"Warning: file {file_number} has no fit data.")
            return x_fit, None
        return x_fit, func(x_fit, *params)

    def get_xy_werror(
        self, file_number, x_fit=None, x_column=None, x_density=1
    ):
        if x_fit is None:
            x, _ = self.get_xy_data(file_number, x_column=x_column)
            x_fit = np.linspace(x[0], x[-1], x_density * len(x))
        func = self.get_scan(file_number).fit_func
        params = self.get_scan(file_number).fit_params
        cov = self.get_scan(file_number).fit_covariance
        if func is None or params is None or cov is None:
            print(f"Warning: file {file_number} has no fit data.")
            return x_fit, None
        stderr = np.sqrt(np.diag(cov))
        y_fit = func(x_fit, *params)
        y_upper = func(x_fit, *(params + stderr))
        y_lower = func(x_fit, *(params - stderr))
        return x_fit, y_fit, y_lower, y_upper


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
            default_sep="\s+",  # noqa
        )

    def normalize_frequency(
        self, *file_numbers, overwrite=False, desired_frequency=10
    ):  # noqa
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
            default_sep="\t",
        )


def loadtest():
    """
    Loads a basic experiment into 'test_exp' for testing purposes.
    """
    test_exp = Experiment()
    x = np.linspace(0, 100, 1000)
    y1 = (
        5 / (1 + (x - 15) ** 2)
        + 5 / (1 + (x - 15.5) ** 2)
        - 30 / (4 + (x - 40) ** 2)
    )
    y2 = x
    test_exp.add_scan(
        filename=(
            r"/path/to/your/file"
            r"/Written_8.83V_and_stuff_abo6.6 GHzdomness.ascii"
        ),
        info=[
            "Current (A): 5\n",
            "Voltage (V): 13\n",
            "Field (G): 3500.23434\n",
            "Sweep Direction: Up\n",
            "Temperature(K): 273\n",
        ],
        axes=["Ax0", "Ax1", "Ax2"],
        data=np.array([x, y1, y2]),
    )
    test_exp.add_scan(
        filename=(
            r"/path/to/your/file"
            r"/Written_4.77V with 9.83 GHz and whatever.txt"
        ),
        info=[
            "Current (A): 5.5\n",
            "Voltage (V): 42\n",
            "Field (G): 3526.4\n",
            "Sweep Direction: Down\n",
            "Temperature(K): 77\n",
        ],
        axes=["Ax0", "Ax1", "Ax2"],
        data=np.array([x, y1, y2]),
    )
    return test_exp
