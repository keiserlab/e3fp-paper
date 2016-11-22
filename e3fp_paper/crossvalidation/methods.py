"""Various classes for cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import sys
import math
import cPickle as pkl

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn import svm
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import leaky_rectify, softmax
from nolearn.lasagne import NeuralNet, BatchIterator
from python_utilities.io_tools import smart_open, touch_dir
from ..sea_utils.util import molecules_to_lists_dicts
from ..sea_utils.library import build_library
from ..sea_utils.run import sea_set_search

RANDOM_STATE = 42
MIN_PVALUE_EXPONENT = math.log10(sys.float_info.epsilon * sys.float_info.min)


class CVMethod(object):

    """Base class for running cross-validation."""

    default_pred = 0.  # Default value if pair missing from results
    fit_file_ext = ".csv.bz2"

    def __init__(self, out_dir="", overwrite=False):
        """Initialize object.

        Parameters
        ----------
        out_dir : str, optional
            Directory in which to save any output files during initialization.
        overwrite : bool, optional
            Overwrite any output files if they already exist.
        """
        self.out_dir = out_dir
        self.overwrite = overwrite

    def train(self, fp_array, mol_to_fp_inds, target_mol_array,
              target_list, mol_list, mask):
        """Train the model.

        Parameters
        ----------
        fp_array : ndarray or csr_matrix (n_fprints, n_bits) or None
            Array with fingerprints as rows
        mol_to_fp_inds : dict
            Map from `mol_name` to indices for mol fingerprints in `fp_array`
        target_mol_array : ndarray of bool (n_targets, n_mols)
            Boolean array with True marking mol/target binding pairs
            and False marking implied negatives.
        target_list : list of str
            List of target names corresponding to rows of `target_mol_array`.
        mol_list : list of str
            List of mol names corresponding to columns of `target_mol_array`.
        mask : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mol/target pairs in the
            training dataset.
        """
        if self.is_trained(target_list) and self.overwrite:
            logging.warning("Re-training model for cross-validation.")

    def test(self, fp_array, mol_to_fp_inds, target_mol_array, target_list,
             mol_list, mask):
        """Score test molecules against using trained model.

        A high score should correspond to a more positive prediction.

        Parameters
        ----------
        fp_array : ndarray or csr_matrix (n_fprints, n_bits) or None
            Array with fingerprints as rows
        mol_to_fp_inds : dict
            Map from `mol_name` to indices for mol fingerprints in `fp_array`
        target_mol_array : ndarray of bool (n_targets, n_mols)
            Boolean array with True marking mol/target binding pairs
            and False marking implied negatives.
        target_list : list of str
            List of target names corresponding to rows of `target_mol_array`.
        mol_list : list of str
            List of mol names corresponding to columns of `target_mol_array`.
        mask : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mol/target pairs in the
            test dataset.

        Returns
        -------
        ndarray of float64 (n_targets, n_mols)
            Array of scores for target, mol pairs.
        """
        return None

    def is_trained(self, target_list=[]):
        """Model has been trained.

        Parameters
        ----------
        target_list : list of TargetKey, optional
            List of target keys for which to check if a model has been
            trained.
        """
        return False

    def _fit_file_from_target_key(self, target_key):
        """Get filename for target fit file."""
        return os.path.join(self.fit_dir, target_key.tid + self.fit_file_ext)


class RandomCVMethod(CVMethod):

    """Cross-validation method using a random classifier.

       This class is provided for testing and to develop baseline ROC and
       Precision-Recall curves/AUCs. It should not be used for actual
       classification.
    """

    def test(self, fp_array, mol_to_fp_inds, target_mol_array, target_list,
             mol_list, mask):
        return np.random.uniform(size=(len(target_list), len(mol_list)))


class SEASearchCVMethod(CVMethod):

    """Cross-validation method using the Similarity Ensemble Approach."""

    fit_file_ext = ".sea"

    def __init__(self, out_dir="", overwrite=False):
        super(SEASearchCVMethod, self).__init__(out_dir=out_dir)
        self.library_file = os.path.join(self.out_dir, "library.sea")
        self.fit_file = os.path.join(self.out_dir, "library.fit")
        self.train_molecules_file = os.path.join(
            self.out_dir, "train_molecules.csv.bz2")
        self.train_targets_file = os.path.join(
            self.out_dir, "train_targets.csv.bz2")
        self.test_molecules_file = os.path.join(
            self.out_dir, "test_molecules.csv.bz2")
        self.test_targets_file = os.path.join(
            self.out_dir, "test_targets.csv.bz2")

    def train(self, fp_array, mol_to_fp_inds, target_mol_array,
              target_list, mol_list, mask):
        """Determine significance threshold and build SEA library.

        Parameters
        ----------
        fp_array : ndarray or csr_matrix (n_fprints, n_bits) or None
            Array with fingerprints as rows
        mol_to_fp_inds : dict
            Map from index of `mol_list` to indices for mol fingerprints in
            `fp_array`
        target_mol_array : ndarray of bool (n_targets, n_mols)
            Boolean array with True marking mol/target binding pairs
            and False marking implied negatives.
        target_list : list of str
            List of target names corresponding to rows of `target_mol_array`.
        mol_list : list of str
            List of mol names corresponding to columns of `target_mol_array`.
        mask : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mol/target pairs in the
            training dataset.
        """
        if self.is_trained(target_list) and not self.overwrite:
            logging.debug("All libraries already have been built.")

        logging.info("Fitting background distribution.")
        if (self.overwrite or not os.path.isfile(self.library_file) or
                              not os.path.isfile(self.fit_file)):
            build_library(self.library_file, self.train_molecules_file,
                          self.train_targets_file, self.fit_file,
                          generate_fit=True)

    def test(self, fp_array, mol_to_fp_inds, target_mol_array, target_list,
             mol_list, mask):
        """Search test molecules against training targets using SEA.

        Parameters
        ----------
        fp_array : ndarray or csr_matrix (n_fprints, n_bits) or None
            Array with fingerprints as rows
        mol_to_fp_inds : dict
            Map from index of `mol_list` to indices for mol fingerprints in
            `fp_array`
        target_mol_array : ndarray of bool (n_targets, n_mols)
            Boolean array with True marking mol/target binding pairs
            and False marking implied negatives.
        target_list : list of str
            List of target names corresponding to rows of `target_mol_array`.
        mol_list : list of str
            List of mol names corresponding to columns of `target_mol_array`.
        mask : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mol/target pairs in the
            test dataset.

        Returns
        -------
        ndarray of float64 (n_targets, n_mols)
            Array of -log10(p-value) for target, mol pairs.
        """
        smiles_dict, mol_list_dict, fp_type = molecules_to_lists_dicts(
            self.test_molecules_file)

        results = np.ones(shape=(len(target_list), len(mol_list)),
                          dtype=np.float64) * self.default_pred
        mol_inds = {x: i for i, x in enumerate(mol_list)}
        target_inds = {x: i for i, x in enumerate(target_list)}

        # search sequences against library
        mol_target_map = {}
        for i, j in zip(*np.where(mask)):
            target_key = target_list[i]
            mol_name = mol_list[j]
            mol_target_map.setdefault(mol_name, set()).add(target_key)

        searcher = sea_set_search(self.library_file, mol_list_dict,
                                  mol_target_map=mol_target_map, log=False)
        results_dict = searcher.set_results_dict

        # Convert p-values to -log10(p-value)
        for mol_name, mol_results in results_dict.iteritems():
            j = mol_inds[mol_name]
            for target_key, pred in mol_results.iteritems():
                i = target_inds[target_key]
                results[i, j] = self.pvalue_to_neglog10p(pred[0])

        return results

    def is_trained(self, target_list):
        if not (os.path.isfile(self.fit_file) and
                os.path.isfile(self.library_file)):
            return False
        return True

    @staticmethod
    def pvalue_to_neglog10p(pvalue):
        """Get -log10(`pvalue`). Set to system max if `pvalue` rounds to 0.

           If `pvalue` is too low, it rounds to 0. Because -log10(`pvalue`)
           will be used for the threshold, these are set to a value higher than
           higher than the highest -log10(`pvalue`).
        """
        if pvalue == 0.:
            return -MIN_PVALUE_EXPONENT + 1.  # Return greater than max.
        return -math.log10(pvalue)


class ClassifierCVMethodBase(CVMethod):

    """Base class for defining classifier methods."""

    dtype = np.float64
    dense_data = False
    fit_file_ext = ".pkl.bz2"
    train_batch_size = None
    test_batch_size = None
    train_sample_negatives = False

    def __init__(self, *args, **kwargs):
        super(ClassifierCVMethodBase, self).__init__(*args, **kwargs)
        self.fit_dir = os.path.join(self.out_dir, "target_fits")
        touch_dir(self.fit_dir)

    @staticmethod
    def create_clf(data=None):
        """Initialize new classifier."""
        raise NotImplementedError

    @staticmethod
    def train_clf(self, data, result, batch_size=None):
        """Train classifier with data and result, optionally in batches."""
        raise NotImplementedError

    @staticmethod
    def score_clf(clf, data, result):
        """Score trained classifier."""
        raise NotImplementedError

    @staticmethod
    def calculate_pred(clf, data):
        """Compute probabilities of positive for dataset."""
        raise NotImplementedError

    def is_trained(self, target_keys=[]):
        """Check if target models are trained."""
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
            pkl.dump(clf, f)
        return fit_file

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        with smart_open(fit_file, "r") as f:
            clf = pkl.load(f)
        return clf

    def get_fprint_subsets(self, mol_to_fp_inds, target_mol_row,
                           mask_row, sample_negatives=False):
        """Get indices for subset of fingerprints for a target.

        Parameters
        ----------
        mol_to_fp_inds : dict
            Map from index of `mol_list` to indices for mol fingerprints in
            `fp_array`
        target_mol_row : ndarray of bool (n_mols,)
            Boolean array with True marking mols that bind to target and False
            marking implied negatives.
        mask_row : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mols in the dataset for the
            target.
        sample_negatives : bool, optional
            To balance classes, sample from negatives to to produce the same
            number of positives.

        Returns
        -------
        fp_inds : list (n_subset_fprints,)
            List of indices for subset of fingerprints in the dataset which
            will be use for testing/training.
        mol_inds : list (n_subset_mols,)
            List indicating whether a fingerprint is a positive or a negative
            with respect to the target.
        mol_fp_num : list (n_subset_mols,)
            List indicating number of fingerprints for each mol.
        """
        if sample_negatives:
            pos_mol_inds = np.where(mask_row & target_mol_row)[0]
            neg_mol_inds = np.where(mask_row & ~target_mol_row)[0]
            neg_mol_inds = np.random.choice(neg_mol_inds, len(pos_mol_inds),
                                            replace=False)
            mol_inds = np.concatenate([pos_mol_inds, neg_mol_inds])
        else:
            mol_inds = np.where(mask_row)[0]
        fp_inds, mol_fp_num = zip(*[(mol_to_fp_inds[i], len(mol_to_fp_inds[i]))
                                    for i in mol_inds])
        fp_inds = [y for x in fp_inds for y in x]
        return fp_inds, list(mol_inds), mol_fp_num

    def train(self, fp_array, mol_to_fp_inds, target_mol_array, target_list,
              mol_list, mask):
        """Train and score a classifier for each target.

        Parameters
        ----------
        fp_array : ndarray or csr_matrix (n_fprints, n_bits)
            Array with fingerprints as rows
        mol_to_fp_inds : dict
            Map from index of `mol_list` to indices for mol fingerprints in
            `fp_array`
        target_mol_array : ndarray of bool (n_targets, n_mols)
            Boolean array with True marking mol/target binding pairs
            and False marking implied negatives.
        target_list : list of str
            List of target names corresponding to rows of `target_mol_array`.
        mol_list : list of str
            List of mol names corresponding to columns of `target_mol_array`.
        mask : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mol/target pairs in the
            training dataset.
        """
        if self.is_trained(target_list) and not self.overwrite:
            logging.info("All targets already trained.")
            return

        if self.dense_data and issparse(fp_array):
            logging.info("Converting from sparse to dense fingerprints.")
            fp_array = fp_array.toarray()
        fp_array = fp_array.astype(self.dtype)

        logging.info("Generating target fits.")
        target_num = len(target_list)
        target_perc_num = int(target_num / 100)
        for i, target_key in enumerate(target_list):
            fit_file = self._fit_file_from_target_key(target_key)
            if os.path.isfile(fit_file) and not self.overwrite:
                logging.debug(
                    "Fit file for {} already exists. Skipping".format(
                        target_key.tid))
                continue

            # get subset of training data
            set_fp_inds, set_mol_inds, set_fp_num = self.get_fprint_subsets(
                mol_to_fp_inds, target_mol_array[i, :], mask[i, :],
                sample_negatives=self.train_sample_negatives)
            data = fp_array[set_fp_inds, :]
            pos = np.repeat(target_mol_array[i, set_mol_inds],
                            set_fp_num).astype(self.dtype)

            # perform training
            clf = self.create_clf(data)
            logging.debug("Fitting {} using {} fprints ({}/{})".format(
                target_key.tid, data.shape[0], i + 1, target_num))
            self.train_clf(clf, data, pos, batch_size=self.train_batch_size)
            if self.train_sample_negatives:  # expensive if all data used
                score = self.score_clf(clf, data, pos)
                logging.debug("Fitted {} with score {:.4f}. ({}/{})".format(
                    target_key.tid, score, i + 1, target_num))
            else:
                logging.debug("Fitted {}. ({}/{})".format(target_key.tid,
                                                          i + 1, target_num))
            self.save_fit_file(target_key, clf)
            # if (i + 1) % target_perc_num == 0:
            #     logging.info("Fit {:.2f}% of targets ({}/{})".format(
            #         100 * (i + 1) / float(target_num), i + 1, target_num))
        logging.info("Finished fitting targets.")

    def test(self, fp_array, mol_to_fp_inds, target_mol_array, target_list,
             mol_list, mask):
        """Search test molecules against training targets using SEA.

        Parameters
        ----------
        fp_array : ndarray or csr_matrix (n_fprints, n_bits)
            Array with fingerprints as rows
        mol_to_fp_inds : dict
            Map from index of `mol_list` to indices for mol fingerprints in
            `fp_array`
        target_mol_array : ndarray of bool (n_targets, n_mols)
            Boolean array with True marking mol/target binding pairs
            and False marking implied negatives.
        target_list : list of str
            List of target names corresponding to rows of `target_mol_array`.
        mol_list : list of str
            List of mol names corresponding to columns of `target_mol_array`.
        mask : ndarray of bool (n_targets, n_mols)
            Boolean array with positives marking mol/target pairs in the
            test dataset.

        Returns
        -------
        ndarray of float64 (n_targets, n_mols)
            Array of scores for target, mol pairs.
        """
        logging.info("Searching molecules against targets.")
        target_num = len(target_list)
        mol_num = len(mol_list)
        results = np.ones(shape=(target_num, mol_num),
                          dtype=np.float64) * self.default_pred

        for i, target_key in enumerate(target_list):
            fit_file = self._fit_file_from_target_key(target_key)
            if not os.path.isfile(fit_file):
                logging.warning("Target fit file does not exist. Target "
                                "will be skipped during testing.")
                results[i, :] = np.nan
            clf = self.load_fit_file(fit_file)
            # target_ind = target_inds[target_key]
            logging.debug("Searching {} against molecules ({}/{}).".format(
                target_key.tid, i, target_num))

            # get subset of test data
            set_fp_inds, set_mol_inds, set_fp_num = self.get_fprint_subsets(
                mol_to_fp_inds, target_mol_array[i, :], mask[i, :],
                sample_negatives=self.train_sample_negatives)
            data = fp_array[set_fp_inds, :].astype(self.dtype)
            if issparse(data) and self.dense_data:
                data = data.toarray()

            # perform test
            scores = self.calculate_pred(clf, data)
            set_mol_fp_ind_starts = np.r_[0, np.cumsum(set_fp_num)]
            results[i, set_mol_inds] = [
                np.amax(scores[b:e]) for b, e in zip(
                    set_mol_fp_ind_starts[:-1], set_mol_fp_ind_starts[1:])]
        return results

    def _fit_file_from_target_key(self, target_key):
        """Get filename for target."""
        return os.path.join(self.fit_dir, target_key.tid + self.fit_file_ext)


class SKLearnCVMethodBase(ClassifierCVMethodBase):

    """Base class for using scikit-learn based classifiers."""

    @staticmethod
    def train_clf(clf, data, result, batch_size=None):
        """Train classifier with data and result."""
        clf.fit(data, result)

    @staticmethod
    def score_clf(clf, data, result):
        """Score trained classifier."""
        return clf.score(data, result)

    @staticmethod
    def calculate_pred(clf, data):
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

    """Cross-validation method using a Tanimoto Support Vector Machine classifier.

       The resulting SVMs are only feasible if tested with smaller batches.
    """

    default_pred = -np.inf  # max_dist_from_hyperplane_neg
    dense_data = True

    @staticmethod
    def create_clf(data=None):
        """Create SVM classifier with tanimoto kernel."""
        return svm.SVC(kernel=tanimoto_kernel, random_state=RANDOM_STATE)

    @staticmethod
    def calculate_pred(clf, data):
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


class EarlyStopping(object):

    """Terminates training if convergence has been reached.

    From https://github.com/dnouri/kfkd-tutorial"""

    def __init__(self, patience=50):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']

        # Ignore if training loss is greater than valid loss
        if current_train > current_valid:
            return

        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience <= current_epoch:
            logging.debug(("Early stopping. Best valid loss was {:.6f} at "
                           "epoch {}.").format(self.best_valid,
                                               self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


class NeuralNetCVMethod(ClassifierCVMethodBase):

    """Cross-validation method using a Neural Network."""

    dtype = np.int32
    dense_data = True
    fit_file_ext = ".pkl"

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
                      "max_epochs": 1000,
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
    def train_clf(clf, data, result, batch_size=None):
        """Train neural network with data and result."""
        return clf.fit(data, result)

    @staticmethod
    def score_clf(clf, data, result):
        """Score trained neural network."""
        return clf.score(data, result)

    @staticmethod
    def calculate_pred(clf, data):
        """Compute probabilities of positive for dataset."""
        return clf.predict_proba(data)[:, 1]

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        fit_file = self._fit_file_from_target_key(target_key)
        clf.save_params_to(fit_file)

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        clf = self.create_clf()
        clf.load_params_from(fit_file)
        return clf

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
