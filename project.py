#!/usr/bin/env python3

from .experiment import Experiment


class Project:
    """Contain several similar experiments in one place and conduct fits
    and plots among the shared experiments.
    """

    def __init__(self, *file_regex_pairs):
        self.macrofit_func = None
        self.macrofit_params = None
        if file_regex_pairs == ():
            self.experiments = list()
        else:
            self.experiments = [] * len(file_regex_pairs)
        for n, pair in enumerate(file_regex_pairs):
            filename, regex = pair
            self.experiments[n] = Experiment(filename, regex)
