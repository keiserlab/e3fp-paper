"""Various classes for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import sys
import math
import itertools
import cPickle as pkl

import numpy as np
import scipy as sc
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

from python_utilities.io_tools import smart_open
from e3fp_paper.sea_utils.util import molecules_to_lists_dicts, \
                                      targets_to_dict, \
                                      mol_lists_targets_to_targets
from e3fp_paper.sea_utils.library import build_library
from e3fp_paper.sea_utils.run import sea_set_search
from e3fp_paper.pipeline import native_tuple_to_fprint

RANDOM_STATE = 42


class CVMethod(object):

    """Base class for running cross-validation."""

    default_metric = (0.)  # Default value if pair missing from results

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

    def test(self, test_mol_lists_dict):
        """Score test molecules against using trained model.

        A high score should correspond to a more positive prediction.

        Parameters
        ----------
        test_mol_lists_dict : str
            Mol lists dict for test molecules.

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
        super(SEASearchCVMethod, self).__init__(out_dir=out_dir)
        self.library_file = os.path.join(self.out_dir, "library.sea")
        self.fit_file = os.path.join(self.out_dir, "library.fit")
        self.default_metric = (0.0, 0.0)  # (-log(p-value), tc)

    def train(self, molecules_file, targets_file):
        """Determine significance threshold and build SEA library.

        Parameters
        ----------
        molecules_file : str
            SEA format molecules file.
        targets_file : str
            SEA format targets file.
        """
        super(SEASearchCVMethod, self).train(molecules_file, targets_file)
        if self.overwrite or not self.is_trained():
            build_library(self.library_file, molecules_file, targets_file,
                          self.fit_file, generate_fit=True)

    def test(self, test_mol_lists_dict):
        """Compare test molecules against training targets using SEA.

        Parameters
        ----------
        test_mol_lists_dict : str
            Mol lists dict for test molecules.

        Returns
        -------
        dict
            Results of comparison, in format
            {mol_name: {target_key: (-log10(p), tc), ...}, ...}, where
            where -log10(p), where p is the p-value of the molecule-target
            pair, and tc is the max Tanimoto coefficient between a test
            fingerprint and a fingerprint in the training set for that target.
        """
        logging.info("Searching {:d} fingerprints against {}.".format(
            len(test_mol_lists_dict), self.library_file))
        results = sea_set_search(self.library_file, test_mol_lists_dict,
                                 log=True)

        results = results.set_results_dict
        # Convert p-values to -log10(p-value)
        for mol_name in results:
            for target_key, metric in results[mol_name].iteritems():
                results[mol_name][target_key] = (
                    self.pvalue_to_neglog10e(metric[0]), metric[1])

        return results

    def is_trained(self):
        """Library has bin built."""
        return (os.path.isfile(self.fit_file) and
                os.path.isfile(self.library_file))

    @staticmethod
    def pvalue_to_neglog10e(pvalue):
        """Get -log10(p-value); set to system max if p-value rounds to 0."""
        # If e-values are too low, they round to 0. Because -log10(evalue)
        # will be used for the threshold, these are set to a value higher than
        # higher than the highest -log10(evalue).
        if pvalue == 0.:
            return -sys.float_info.min_10_exp + 1.  # Return greater than max.
        return -math.log10(pvalue)


class SKLearnCVMethodBase(CVMethod):

    """Base class for using scikit-learn based classifiers."""

    default_metric = (0.0,)  # (prob,)

    def __init__(self, out_dir="", overwrite=False):
        super(SKLearnCVMethodBase, self).__init__(out_dir=out_dir,
                                                  overwrite=overwrite)
        self.fit_file = os.path.join(self.out_dir, "target_fits.pkl.gz")

    @staticmethod
    def create_clf():
        """Initialize new classifier."""
        raise NotImplementedError

    @staticmethod
    def calculate_metric(clf, data):
        """Compute probabilities of positive for dataset.

        Parameters
        ----------
        clf : sklearn estimator
            Classifier
        data : ndarray or sparse matrix
            NxM array with N data points and M features.
        """
        return clf.predict_proba(data)[:, 1]

    def is_trained(self):
        return os.path.isfile(self.fit_file)

    @staticmethod
    def molecules_to_array(molecules):
        """Convert molecules to sparse matrix.

        Parameters
        ----------
        molecules : dict or string
            Molecules file or mol_list_dict.

        Returns
        -------
        csr_matrix
            Row-based sparse matrix containing fingerprints
        dict
            Map from mol_name to list of row indices in sparse matrix.
        """
        if isinstance(molecules, dict):
            mol_list_dict = molecules
        else:
            _, mol_list_dict, _ = molecules_to_lists_dicts(molecules)

        logging.info("Initializing sparse matrix.")
        fp_num = 0
        mol_indices_dict = {}
        mol_names = sorted(mol_list_dict)
        for mol_name in mol_names:
            mol_fp_num = len(mol_list_dict[mol_name])
            mol_indices_dict[mol_name] = range(fp_num,
                                               fp_num + mol_fp_num)
            fp_num += mol_fp_num

        logging.info("Populating sparse matrix with fingerprints.")
        bit_num = native_tuple_to_fprint(
            next(mol_list_dict.itervalues())[0]).bits
        all_col_inds = []
        all_row_inds = []
        for mol_name, native_tuples in mol_list_dict.iteritems():
            row_inds = mol_indices_dict[mol_name]
            col_inds, row_inds = zip(*[(list(fp.indices),
                                        [r] * fp.bit_count)
                                       for fp, r in zip(map(
                                           native_tuple_to_fprint,
                                           native_tuples), row_inds)])
            all_col_inds.extend(itertools.chain(*col_inds))
            all_row_inds.extend(itertools.chain(*row_inds))
        # csr matrix as np.float64 for quick classification
        all_fps = sc.sparse.coo_matrix(([True] * len(all_row_inds),
                                        (all_row_inds, all_col_inds)),
                                       shape=(fp_num, bit_num),
                                       dtype=np.float64).tocsr()
        del mol_list_dict, all_col_inds, all_row_inds
        return all_fps, mol_indices_dict

    def train(self, molecules_file, targets_file):
        """Train and store a classifier for each target.

        Parameters
        ----------
        molecules_file : str
            SEA format molecules file.
        targets_file : str
            SEA format targets file.
        """
        if self.is_trained() and not self.overwrite:
            return

        logging.info("Loading molecules/targets.")
        all_fps, mol_indices_dict = self.molecules_to_array(molecules_file)
        mol_names_set = set(mol_indices_dict.keys())

        targets_dict = mol_lists_targets_to_targets(
            targets_to_dict(targets_file))

        logging.info("Generating target fits.")
        target_fits = {}
        target_num = len(targets_dict)
        target_perc_num = int(target_num / 100)
        for i, (target_key, set_value) in enumerate(targets_dict.iteritems()):
            pos_mol_names = set(set_value.cids)
            neg_mol_names = set(np.random.choice(
                list(mol_names_set - pos_mol_names), len(pos_mol_names),
                replace=False))
            fp_inds, pos = zip(*([(y, True) for x in pos_mol_names
                                  for y in mol_indices_dict[x]] +
                                 [(y, False) for x in neg_mol_names
                                  for y in mol_indices_dict[x]]))
            data = all_fps[fp_inds, :]
            clf = self.create_clf()
            logging.debug("Fitting {} using {} fprints ({}/{})".format(
                target_key.tid, data.shape[0], i + 1, target_num))
            clf.fit(data, pos)
            score = clf.score(data, pos)
            logging.debug("Fitted {} with score {:.4f} ({}/{})".format(
                target_key.tid, score, i + 1, target_num))
            target_fits[target_key] = clf
            if (i + 1) % target_perc_num == 0:
                logging.info("Fit {:.2f}% of targets ({}/{})".format(
                    100 * (i + 1) / float(target_num), i + 1, target_num))
        del targets_dict, all_fps

        logging.info("Saving target fits.")
        try:
            joblib.dump(target_fits, self.fit_file, compress=9)
        except OverflowError:  # zlib bug in Python 2.7
            with smart_open(self.fit_file, "w") as f:
                pkl.dump(target_fits, f)

    def test(self, test_mol_lists_dict):
        """Compare test molecules against training targets using classifier.

        Parameters
        ----------
        test_mol_lists_dict : str
            Mol lists dict for test molecules.

        Returns
        -------
        dict
            Results of comparison, in format
            {mol_name: {target_key: (metric1, ...), ...}, ...}, where
            where metric1 is the metric used to construct ROC curve.
        """
        logging.info("Loading test molecules.")
        test_fps, test_mol_indices_dict = self.molecules_to_array(
            test_mol_lists_dict)

        logging.info("Loading target fits.")
        try:
            target_fits = joblib.load(self.fit_file)
        except KeyError:
            with smart_open(self.fit_file, "r") as f:
                target_fits = pkl.load(f)

        logging.info("Searching molecules against targets.")
        results = {mol_name: {} for mol_name
                   in test_mol_indices_dict.iterkeys()}
        target_num = len(target_fits)
        for i, (target_key, clf) in enumerate(target_fits.iteritems()):
            logging.debug("Searching {} against molecules ({}/{}).".format(
                target_key.tid, i, target_num))
            scores = self.calculate_metric(clf, test_fps)
            for mol_name, mol_inds in test_mol_indices_dict.iteritems():
                max_score = float(max(scores[mol_inds]))
                results[mol_name][target_key] = (max_score,)
        return results


class SVMCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Support Vector Machine."""

    default_metric = (-np.inf,)  # (max_dist_from_hyperplane_neg,)

    @staticmethod
    def create_clf():
        return svm.SVC(kernel=minmax_kernel, random_state=RANDOM_STATE)

    @staticmethod
    def calculate_metric(clf, data):
        """Compute distances between data points and hyperplane.

        Negative distances are on the "negative" side of the plane, and
        positive on the "positive" side.

        Parameters
        ----------
        clf : sklearn estimator
            Classifier
        data : ndarray or sparse matrix
            NxM array with N data points and M features.
        """
        return clf.decision_function(data)


class RandomForestCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Random Forest."""

    @staticmethod
    def create_clf():
        return RandomForestClassifier(n_estimators=100, max_depth=2,
                                      min_samples_split=2, n_jobs=-1,
                                      random_state=RANDOM_STATE,
                                      class_weight="balanced")


class NaiveBayesCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Naive Bayesian Classifier."""

    @staticmethod
    def create_clf():
        return MultinomialNB(alpha=1.0, fit_prior=True)


def minmax_kernel(X, Y):
    """MinMax kernel for use in kernel methods.

    Abstracts to Tanimoto given binary input data. For bitvector, theoretical
    bounds for any u,v molecule pair would be [0,1].

    Parameters
    ----------
    X : ndarray or csr_matrix of np.float64
        MxP bitvector array for M mols and P bits
    Y : np.array or csr_matrix of np.float64
        NxP bitvector array for N mols and P bits

    Returns
    ----------
    ndarray of np.float64
        Min-Max similarity between X and Y fingerprints

    References
    ----------
    ..[1] L. Ralaivola, S.J. Swamidass, H. Saigo, P. Baldi."Graph kernels for
          chemical informatics." Neural Networks. 2005. 18(8): 1093-1110.
          doi: 10.1.1.92.483
    """
    minkernel = X.dot(Y.transpose())  # bit intersection count
    try:
        X = X.toarray()
    except AttributeError:
        pass
    try:
        Y = Y.toarray()
    except AttributeError:
        pass

    maxkernel = X.shape[1] - np.dot(1 - X, (1 - Y).T)  # bit union count
    with np.errstate(divide='ignore'):  # handle 0 in denominator
        return np.asarray(np.nan_to_num(minkernel / maxkernel))
