"""Various classes for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import glob
import logging
import sys
import math
import cPickle as pkl
import shelve

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn import svm
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import leaky_rectify, softmax
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn_utils.hooks import EarlyStopping
from python_utilities.io_tools import smart_open, touch_dir
from e3fp_paper.sea_utils.util import targets_to_dict, \
                                      mol_lists_targets_to_targets
from e3fp_paper.sea_utils.library import build_library
from e3fp_paper.sea_utils.run import sea_set_search
from e3fp_paper.crossvalidation.util import molecules_to_array, \
                                            save_fprints_arr, load_fprints_arr

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

    def train(self, molecules_file, targets_file, sample=True):
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
        """Library has been built."""
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
    fit_file_ext = ".pkl.gz"

    def __init__(self, out_dir="", overwrite=False):
        super(ClassifierCVMethodBase, self).__init__(out_dir=out_dir,
                                                     overwrite=overwrite)
        self.fit_dir = os.path.join(self.out_dir, "target_fits")
        self.train_fp_file = os.path.join(self.out_dir, "fps_train.npz")
        self.test_fp_file = os.path.join(self.out_dir, "fps_test.npz")

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

    def is_trained(self, target_keys=[]):
        """Check if target models are trained."""
        if not os.path.isdir(self.fit_dir):
            return False
        for target_key in target_keys:
            if not os.path.isfile(self._fit_file_from_target_key(target_key)):
                return False
        return True

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        try:
            fit_file = self._fit_file_from_target_key(target_key)
        except:  # assume target_key is a fit file.
            fit_file = target_key
        with smart_open(fit_file, "w") as f:
            pkl.dump((target_key, clf), f)
        return fit_file

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        try:  # backwards compatibility
            target_key, clf = joblib.load(fit_file)
        except KeyError:
            with smart_open(fit_file, "r") as f:
                target_key, clf = pkl.load(f)
        return target_key, clf

    def train(self, molecules_file, targets_file, sample=True):
        """Train and store a classifier for each target.

        Parameters
        ----------
        molecules_file : str
            SEA format molecules file.
        targets_file : str
            SEA format targets file.
        sample : bool, optional
            Sample negatives to match number of positives to correct for class
            imbalance.
        """
        logging.info("Loading targets for training.")
        targets_dict = mol_lists_targets_to_targets(
            targets_to_dict(targets_file))

        if self.is_trained(targets_dict.keys()) and not self.overwrite:
            logging.info("All targets already trained.")
            return

        if not sample:
            logging.warning("Sample option set to off. All negative and "
                            "positive data will be used for training. This "
                            "is dangerous if classes are imbalanced.")

        logging.info("Loading molecules for training.")
        if (os.path.isfile(self.train_fp_file) and not self.overwrite):
            logging.info("Loading fingerprint array from file.")
            all_fps, mol_indices_dict = load_fprints_arr(
                self.train_fp_file, dense=self.dense_data)
        else:
            all_fps, mol_indices_dict = molecules_to_array(
                molecules_file, dtype=self.dtype, dense=self.dense_data)
            logging.info("Saving fingerprint array to file.")
            save_fprints_arr(self.train_fp_file, all_fps, mol_indices_dict)

        mol_names_set = set(mol_indices_dict.keys())

        logging.info("Generating target fits.")
        touch_dir(self.fit_dir)
        target_num = len(targets_dict)
        target_perc_num = int(target_num / 100)
        for i, (target_key, set_value) in enumerate(targets_dict.iteritems()):
            fit_file = self._fit_file_from_target_key(target_key)
            if os.path.isfile(fit_file) and not self.overwrite:
                logging.debug(
                    "Fit file for {} already exists. Skipping".format(
                        target_key.tid))
                continue

            if sample:
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
            else:
                pos = np.zeros(all_fps.shape[0], dtype=self.dtype)
                pos_mol_names = set(set_value.cids)
                pos[[y for x in pos_mol_names
                     for y in mol_indices_dict[x]]] = 1
                data = all_fps
            clf = self.create_clf(data)
            logging.debug("Fitting {} using {} fprints ({}/{})".format(
                target_key.tid, data.shape[0], i + 1, target_num))
            self.train_clf(clf, data, pos)
            if sample:  # expensive if all data used
                score = self.score_clf(clf, data, pos)
                logging.debug("Fitted {} with score {:.4f}. ({}/{})".format(
                    target_key.tid, score, i + 1, target_num))
            else:
                logging.debug("Fitted {}. ({}/{})".format(target_key.tid,
                                                          i + 1, target_num))
            self.save_fit_file(target_key, clf)
            if (i + 1) % target_perc_num == 0:
                logging.info("Fit {:.2f}% of targets ({}/{})".format(
                    100 * (i + 1) / float(target_num), i + 1, target_num))
        logging.info("Finished fitting targets.")
        del all_fps

    def test(self, test_mol_lists_dict, batch=True):
        """Compare test molecules against training targets using classifier.

        Parameters
        ----------
        test_mol_lists_dict : str
            Mol lists dict for test molecules.
        batch : bool, optional
            Predict all fingerprints with classifier simultaneously. If False,
            each molecule will be predicted separately.

        Returns
        -------
        dict
            Results of comparison, in format
            {mol_name: {target_key: (metric1, ...), ...}, ...}, where
            where metric1 is the metric used to construct ROC curve.
        """
        logging.info("Loading molecules for testing.")
        if (os.path.isfile(self.test_fp_file) and not self.overwrite):
            logging.info("Loading fingerprint array from file.")
            test_fps, test_mol_indices_dict = load_fprints_arr(
                self.test_fp_file, dense=self.dense_data)
        else:
            test_fps, test_mol_indices_dict = molecules_to_array(
                test_mol_lists_dict, dtype=self.dtype, dense=self.dense_data)
            logging.info("Saving fingerprint array to file.")
            save_fprints_arr(self.test_fp_file, test_fps,
                             test_mol_indices_dict)

        logging.info("Fetching target fits.")
        fit_files = glob.glob(os.path.join(self.fit_dir, "*" + self.fit_file_ext))

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

    def _fit_file_from_target_key(self, target_key):
        """Get filename for target."""
        return os.path.join(self.fit_dir, target_key.tid + self.fit_file_ext)


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


class RandomCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a random classifier.

       This class is provided to develop baseline ROC and Precision-Recall
       curves/AUCs. It should not be used for actual classification.
    """

    @staticmethod
    def create_clf(data=None):
        """Create random classifier."""
        return DummyClassifier(strategy="uniform")


class SVMCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Tanimoto Support Vector Machine classifier.

       The resulting SVMs are only feasible if tested with smaller batches.
    """

    default_metric = (-np.inf,)  # (max_dist_from_hyperplane_neg,)
    dense_data = True

    @staticmethod
    def create_clf(data=None):
        """Create SVM classifier with tanimoto kernel."""
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


class LinearSVMCVMethod(SVMCVMethod):

    """Cross-validation method using a linear Support Vector Machine classifier.

       Parameters are chosen to optimize speed/memory for sparse arrays.
    """

    dense_data = False

    @staticmethod
    def create_clf(data=None):
        """Create SVM classifier with linear kernel."""
        return svm.LinearSVC(penalty="l1", dual=False,
                             class_weight="balanced", loss="squared_hinge",
                             tol=1e-4, random_state=RANDOM_STATE)

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        clf.sparsify()
        return super(LinearSVMCVMethod, self).save_fit_file(target_key, clf)


class RandomForestCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Random Forest."""

    @staticmethod
    def create_clf(data=None):
        """Create Random Forest classifier."""
        return RandomForestClassifier(n_estimators=100, max_depth=2,
                                      min_samples_split=2, n_jobs=-1,
                                      random_state=RANDOM_STATE,
                                      class_weight="balanced")


class NaiveBayesCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Naive Bayesian Classifier."""

    @staticmethod
    def create_clf(data=None):
        return BernoulliNB(alpha=1.0, fit_prior=True)


class BalancedClassIterator(BatchIterator):

    """Iterator that re-samples until both classes have equal probabilities.

       Subsample data to correct for class imbalance by sampling. The epoch is
       defined as the amount of sampled data equal to double the size of the
       smallest class.
    """

    def __init__(self, *args, **kwargs):
        kwargs["shuffle"] = False
        self.pos_inds = None
        self.neg_inds = None
        self.min_count = 0
        super(BalancedClassIterator, self).__init__(*args, **kwargs)

    def __call__(self, X, y=None):
        if y is not None:
            if self.pos_inds is None or self.neg_inds is None:
                neg_inds = y == 0
                self.pos_inds = np.where(~neg_inds)[0]
                self.neg_inds = np.where(neg_inds)[0]
                self.min_count = min(self.neg_inds.shape[0],
                                     self.pos_inds.shape[0])
            pos_inds = np.random.choice(self.pos_inds,
                                        size=self.min_count,
                                        replace=True)
            neg_inds = np.random.choice(self.neg_inds,
                                        size=self.min_count,
                                        replace=True)
            rand_inds = np.concatenate((pos_inds, neg_inds))
            X, y = X[rand_inds], y[rand_inds]
        return super(BalancedClassIterator, self).__call__(X, y)


class NeuralNetCVMethod(ClassifierCVMethodBase):

    """Cross-validation method using a Neural Network."""

    dtype = np.int32
    dense_data = True
    fit_file_ext = ".pkl"

    def __init__(self, *args, **kwargs):
        super(NeuralNetCVMethod, self).__init__(*args, **kwargs)
        self.target_fits = shelve.open(os.path.join(self.out_dir,
                                                    'target_files_key.db'))

    @staticmethod
    def create_clf(data=None):
        """Create neural network."""
        net_params = {"layers": [("input", InputLayer),
                                 ("inputdrop", DropoutLayer),
                                 ("hidden", DenseLayer),
                                 ("hiddendrop", DropoutLayer),
                                 ("output", DenseLayer)],
                      "input_shape": (None, 1024),
                      "inputdrop_p": .1,
                      "hidden_num_units": 512,
                      "hidden_nonlinearity": leaky_rectify,
                      "hiddendrop_p": .25,
                      "output_num_units": 2,
                      "output_nonlinearity": softmax,
                      "update_learning_rate": 0.01,
                      "max_epochs": 300,
                      "on_epoch_finished": EarlyStopping(patience=75)}
        clf = NeuralNet(**net_params)
        if data is not None:
            batch_size = min(1000, int(.2 * data.shape[0]))
            clf.batch_iterator_train = BalancedClassIterator(
                batch_size=batch_size)
            clf.batch_iterator_test = BalancedClassIterator(
                batch_size=batch_size)

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
        fit_file = self._fit_file_from_target_key(target_key)
        self.target_fits[fit_file] = target_key
        clf.save_params_to(fit_file)

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        target_key = self.target_fits[fit_file]
        clf = self.create_clf()
        clf.load_params_from(fit_file)
        return target_key, clf

    def train(self, molecules_file, targets_file, sample=False):
        super(NeuralNetCVMethod, self).train(molecules_file, targets_file,
                                             sample=sample)

    def __del__(self):
        self.target_fits.close()
        del self.target_fits


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
