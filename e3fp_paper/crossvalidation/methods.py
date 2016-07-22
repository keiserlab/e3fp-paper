"""Various classes for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging

from e3fp.sea_utils.library import build_library
from e3fp.sea_utils.run import sea_set_search


class CVMethod(object):

    """Base class for running cross-validation."""

    def __init__(self, out_dir="", overwrite=False):
        """Constructor.

        Parameters
        ----------
        out_dir : str, optional
            Directory in which to save any output files during initialization.
        overwrite : bool, optional
            Overwrite any output files if they already exist.
        """
        self.out_dir = out_dir
        self.overwrite = overwrite
        self.default_metric = tuple(0.)  # Default value if pair missing from
                                         # results
        self.order = "greater"  # A greater value of the above metric is
                                # better

    def train(self, molecules_file, targets_file):
        """Train the model.

        Parameters
        ----------
        molecules_file : str
            SEA format molecules file.
        targets_file : str
            SEA format targets file.
        """
        if self.is_trained() and self.overwrite:
            logging.warning("Re-training model for cross-validation.")

    def compare(self, test_mol_lists_dict, train_molecules_file,
                train_targets_file, cv_dir=""):
        """Summary

        Parameters
        ----------
        test_mol_lists_dict : str
            Mol lists dict for test molecules.
        train_molecules_file : str
            SEA format molecules file for training molecules.
        train_targets_file : str
            SEA format targets file for training molecules.
        cv_dir : str, optional
            Directory in which to save any output files.

        Returns
        -------
        dict
            Results of comparison, in format
            {mol_name: {target_key: (metric1, ...), ...}, ...}, where
            where metric1 is the metric used to construct ROC curve.
        """
        return {}

    def is_trained(self):
        """Model has been trained."""
        return False


class SEASearchCVMethod(CVMethod):

    def __init__(self, out_dir="", overwrite=False):
        super(CVMethod, self).__init__(out_dir=out_dir)
        self.library_file = os.path.join(self.out_dir, "library.fit")
        self.fit_file = os.path.join(self.out_dir, "library.sea")
        self.default_metric = (1.0, 0.0)  # (p-value, tc)
        self.order = "less"  # A smaller value of the above metric is better

    def train(self, molecules_file, targets_file, generate_fit=True):
        super(CVMethod, self).train(molecules_file, targets_file)
        if os.path.isfile(self.fit_file) and not self.overwrite:
            logging.warning("Fit file already exists. Will not generate fit.")
            generate_fit = False

        if self.overwrite or not self.is_trained():
            build_library(self.library_file, molecules_file, targets_file,
                          self.fit_file, generate_fit=generate_fit)

    def compare(self, test_mol_lists_dict, train_molecules_file,
                train_targets_file, cv_dir=None):
        if cv_dir is None:
            cv_dir = self.out_dir
        train_library_file = os.path.join(cv_dir, "library.sea")

        if self.overwrite or not os.path.isfile(train_library_file):
            logging.info("Building library for training set.")
            build_library(train_library_file, train_molecules_file,
                          train_targets_file, self.fit_file,
                          generate_fit=False)

        logging.info("Searching {:d} fingerprints against {}.".format(
            len(test_mol_lists_dict), train_library_file))
        results = sea_set_search(train_library_file, test_mol_lists_dict,
                                 log=True)

        return results.set_results_dict

    def is_trained(self):
        return (os.path.isfile(self.fit_file) and
                os.path.isfile(self.library_file))
