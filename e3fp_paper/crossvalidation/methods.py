"""Various classes for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import logging
import sys
import math
import itertools
import cPickle as pkl

import numpy as np
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sklearn import svm
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.externals import joblib
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import leaky_rectify, softmax
from nolearn.lasagne import NeuralNet, BatchIterator
from python_utilities.io_tools import smart_open, touch_dir
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


class ClassifierCVMethodBase(CVMethod):

    """Base class for defining classifier methods."""

    default_metric = (0.0,)  # (prob,)
    dtype = np.float64
    dense_data = False

    def __init__(self, out_dir="", overwrite=False):
        super(ClassifierCVMethodBase, self).__init__(out_dir=out_dir,
                                                     overwrite=overwrite)
        self.fit_dir = os.path.join(self.out_dir, "target_fits")

    @staticmethod
    def create_clf(data=None):
        """Initialize new classifier."""
        raise NotImplementedError

    @staticmethod
    def train_clf(clf, data, result):
        """Train classifier with data and result."""
        raise NotImplementedError

    @staticmethod
    def score_clf(clf, data, result):
        """Score trained classifier."""
        raise NotImplementedError

    @staticmethod
    def calculate_metric(clf, data):
        """Compute probabilities of positive for dataset."""
        raise NotImplementedError

    def is_trained(self):
        return (os.path.isdir(self.fit_dir) and
                len(glob.glob(os.path.join(self.fit_dir, "*"))) > 0)

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        fit_file = os.path.join(self.fit_dir, target_key.tid + ".pkl.gz")
        try:
            joblib.dump((target_key, clf), fit_file, compress=9)
        except OverflowError:  # zlib bug in Python 2.7
            with smart_open(fit_file, "w") as f:
                pkl.dump((target_key, clf), f)
        return fit_file

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        try:
            target_key, clf = joblib.load(fit_file)
        except KeyError:
            with smart_open(fit_file, "r") as f:
                target_key, clf = pkl.load(f)
        return target_key, clf

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
        all_fps, mol_indices_dict = molecules_to_array(molecules_file,
                                                       dtype=self.dtype,
                                                       dense=self.dense_data)
        mol_names_set = set(mol_indices_dict.keys())

        targets_dict = mol_lists_targets_to_targets(
            targets_to_dict(targets_file))

        logging.info("Generating target fits.")
        touch_dir(self.fit_dir)
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
            pos = np.asarray(pos, dtype=self.dtype)
            data = all_fps[fp_inds, :]
            clf = self.create_clf(data)
            logging.debug("Fitting {} using {} fprints ({}/{})".format(
                target_key.tid, data.shape[0], i + 1, target_num))
            self.train_clf(clf, data, pos)
            score = self.score_clf(clf, data, pos)
            logging.debug("Fitted {} with score {:.4f}. ({}/{})".format(
                target_key.tid, score, i + 1, target_num))
            self.save_fit_file(target_key, clf)
            if (i + 1) % target_perc_num == 0:
                logging.info("Fit {:.2f}% of targets ({}/{})".format(
                    100 * (i + 1) / float(target_num), i + 1, target_num))
        del all_fps

    def test(self, test_mol_lists_dict, batch=True):
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
        test_fps, test_mol_indices_dict = molecules_to_array(
            test_mol_lists_dict, dtype=self.dtype, dense=self.dense_data)

        logging.info("Fetching target fits.")
        fit_files = glob.glob(os.path.join(self.fit_dir, "*"))

        logging.info("Searching molecules against targets.")
        results = {mol_name: {} for mol_name
                   in test_mol_indices_dict.iterkeys()}
        target_num = len(fit_files)
        for i, fit_file in enumerate(fit_files):
            target_key, clf = self.load_fit_file(fit_file)
            logging.debug("Searching {} against molecules ({}/{}).".format(
                target_key.tid, i, target_num))
            if batch:
                scores = self.calculate_metric(clf, test_fps)
                for mol_name, mol_inds in test_mol_indices_dict.iteritems():
                    max_score = float(max(scores[mol_inds]))
                    results[mol_name][target_key] = (max_score,)
            else:
                for mol_name, mol_inds in test_mol_indices_dict.iteritems():
                    scores = self.calculate_metric(clf, test_fps[mol_inds, :])
                    max_score = float(max(scores))
                    results[mol_name][target_key] = (max_score,)
        return results


def molecules_to_array(molecules, dtype=np.float64, dense=False):
    """Convert molecules to array or sparse matrix.

    Parameters
    ----------
    molecules : dict or string
        Molecules file or mol_list_dict.
    dtype : type, optional
        Numpy data type.
    dense : bool, optional
        Return dense array.

    Returns
    -------
    csr_matrix or ndarray
        Row-based sparse matrix or ndarray containing fingerprints.
    dict
        Map from mol_name to list of row indices of fingerprints.
    """
    if isinstance(molecules, dict):
        mol_list_dict = molecules
    else:
        _, mol_list_dict, _ = molecules_to_lists_dicts(molecules)

    fp_num = 0
    mol_indices_dict = {}
    mol_names = sorted(mol_list_dict)
    for mol_name in mol_names:
        mol_fp_num = len(mol_list_dict[mol_name])
        mol_indices_dict[mol_name] = range(fp_num,
                                           fp_num + mol_fp_num)
        fp_num += mol_fp_num

    bit_num = native_tuple_to_fprint(
        next(mol_list_dict.itervalues())[0]).bits
    if dense:
        logging.info("Populating array with fingerprints.")
        all_fps = np.empty((fp_num, bit_num), dtype=dtype)
        for mol_name, native_tuples in mol_list_dict.iteritems():
            row_inds = mol_indices_dict[mol_name]
            all_fps[row_inds, :] = [
                native_tuple_to_fprint(n).to_bitvector()
                for n in native_tuples]
    else:
        logging.info("Populating sparse matrix with fingerprints.")
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

        all_fps = coo_matrix(([True] * len(all_row_inds),
                              (all_row_inds, all_col_inds)),
                             shape=(fp_num, bit_num),
                             dtype=dtype).tocsr()
        del all_col_inds, all_row_inds

    del mol_list_dict
    return all_fps, mol_indices_dict


class SKLearnCVMethodBase(ClassifierCVMethodBase):

    """Base class for using scikit-learn based classifiers."""

    @staticmethod
    def train_clf(clf, data, result):
        """Train classifier with data and result."""
        clf.fit(data, result)

    @staticmethod
    def score_clf(clf, data, result):
        """Score trained classifier."""
        return clf.score(data, result)

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


class SVMCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Support Vector Machine."""

    default_metric = (-np.inf,)  # (max_dist_from_hyperplane_neg,)
    dense_data = True

    @staticmethod
    def create_clf(data=None):
        return svm.SVC(kernel=tanimoto_kernel, random_state=RANDOM_STATE)

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
    def create_clf(data=None):
        return RandomForestClassifier(n_estimators=100, max_depth=2,
                                      min_samples_split=2, n_jobs=-1,
                                      random_state=RANDOM_STATE,
                                      class_weight="balanced")


class NaiveBayesCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Naive Bayesian Classifier."""

    @staticmethod
    def create_clf(data=None):
        return BernoulliNB(alpha=1.0, fit_prior=True)


class NeuralNetCVMethod(ClassifierCVMethodBase):

    """Cross-validation method using a Neural Network."""

    dtype = np.int32
    dense_data = True
    target_fits = {}

    @staticmethod
    def create_clf(data=None):
        net_params = {"layers": [("input", InputLayer),
                                 ("inputdrop", DropoutLayer),
                                 ("hidden", DenseLayer),
                                 ("hiddendrop", DropoutLayer),
                                 ("output", DenseLayer)],
                      "input_shape": (None, 1024),
                      "inputdrop_p": .25,
                      "hidden_num_units": 512,
                      "hidden_nonlinearity": leaky_rectify,
                      "hiddendrop_p": .1,
                      "output_num_units": 2,
                      "output_nonlinearity": softmax,
                      "update_learning_rate": 0.01,
                      "max_epochs": 100}
        clf = NeuralNet(**net_params)
        if data is not None:
            batch_size = min(1000, int(.2 * data.shape[0]))
            clf.batch_iterator_train = BatchIterator(batch_size=batch_size)
        return clf

    @staticmethod
    def train_clf(clf, data, result):
        """Train neural network with data and result."""
        return clf.fit(data, result)

    @staticmethod
    def score_clf(clf, data, result):
        """Score trained neural network."""
        return clf.score(data, result)

    @staticmethod
    def calculate_metric(clf, data):
        """Compute probabilities of positive for dataset."""
        return clf.predict_proba(data)[:, 1]

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        fit_file = os.path.join(self.fit_dir, target_key.tid + ".pkl")
        self.target_fits[fit_file] = target_key
        clf.save_params_to(fit_file)

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        target_key = self.target_fits[fit_file]
        clf = self.create_clf()
        clf.load_params_from(fit_file)
        return target_key, clf


def tanimoto_kernel(X, Y=None):
    """Compute the Tanimoto kernel between X and Y.

    Data must be binary. This is not checked.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).
    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    similarity_matrix : array of shape (n_samples_X, n_samples_Y)

    References
    ----------
    ..[1] L. Ralaivola, S.J. Swamidass, H. Saigo, P. Baldi."Graph kernels for
          chemical informatics." Neural Networks. 2005. 18(8): 1093-1110.
          doi: 10.1.1.92.483
    """
    X, Y = check_pairwise_arrays(X, Y)
    if issparse(X) or issparse(Y):  # ensure if one is sparse, all are sparse.
        X = csr_matrix(X, copy=False)
        Y = csr_matrix(Y, copy=False)
    Xbits = np.sum(X, axis=1, keepdims=True)
    Ybits = np.sum(Y, axis=1, keepdims=True)
    XYbits = safe_sparse_dot(X, Y.T, dense_output=True)
    with np.errstate(divide='ignore'):  # handle 0 in denominator
        return np.asarray(np.nan_to_num(XYbits / (Xbits + Ybits.T - XYbits)))
