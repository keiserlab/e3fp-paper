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
from scipy.sparse import issparse
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import leaky_rectify, softmax
from nolearn.lasagne import NeuralNet, BatchIterator
from python_utilities.io_tools import smart_open, touch_dir
from ..sea_utils.util import molecules_to_lists_dicts
from ..sea_utils.library import build_library
from ..sea_utils.run import sea_set_search
from e3fp.fingerprint.metrics.array_metrics import tanimoto

RANDOM_STATE = 42
MIN_PVALUE_EXPONENT = math.log10(sys.float_info.epsilon * sys.float_info.min)
RESULTS_DTYPE = np.float64


class CVMethod(object):

    """Base class for running cross-validation.

    Attributes
    ----------
    out_dir : str
        Directory in which to save any output files.
    overwrite : bool
        Overwrite any output files if they already exist.
    """

    default_pred = 0.  # Default value if pair missing from results
    fit_file_ext = ".csv.bz2"

    def __init__(self):
        """Initialize object."""
        self.out_dir = "./"
        self.overwrite = False

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out_dir):
        self._out_dir = out_dir

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


class ScoreMatrix(object):

    """Easily access members of a memory-mapped symmetric score matrix."""

    def __init__(self, memmap_file, entry_names_file, perfect_score=1.,
                 dtype=np.double):
        self.memmap_file = memmap_file
        self.array = None
        self.dtype = dtype
        self.entry_names_file = entry_names_file
        self.entry_names = []
        self.name_to_index_map = {}
        self.perfect_score = 1.
        self.shape = (0, 0)

    def load(self):
        """Load memmap file and entry names file."""
        self.array = np.memmap(self.memmap_file, mode="r", dtype=self.dtype)
        self.entry_names = []
        with smart_open(self.entry_names_file, "r") as f:
            for line in f:
                line = line.rstrip()
                if len(line) > 0:
                    self.entry_names.append(line)
        size = len(self.entry_names)
        if self._get_tril_index_from_indices(
                size - 1, size - 2) != self.array.shape[0] - 1:
            raise ValueError(("Number of items in memmap does not match "
                              "number of row names."))
        self.shape = (size, size)
        self.update_name_to_index_map()

    def is_loaded(self):
        return self.array is not None

    def update_name_to_index_map(self):
        self.name_to_index_map = {}
        for i, entry_name in enumerate(self.entry_names):
            self.name_to_index_map[entry_name] = i
        if len(self.name_to_index_map) != len(self.entry_names):
            raise ValueError("Not all row names are unique.")

    @staticmethod
    def _get_tril_index_from_indices(i, j):
        """Map symmetric matrix indices to flat lower triangle indices."""
        i, j = np.atleast_2d(i).T, np.atleast_2d(j)
        imat, jmat = np.maximum(i, j), np.minimum(i, j)
        indices = imat * (imat - 1) // 2 + jmat
        indices[imat - jmat == 0] = -1
        return indices

    def _key_to_indices(self, key):
        """Convert key or list of keys to array indices."""
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return self.name_to_index_map[key]
        else:
            return [self.name_to_index_map.get(x, x) for x in key]

    def __getitem__(self, key):
        """Get items by row/col name pair."""
        if not self.is_loaded():
            raise KeyError("Array has not yet been loaded.")
        key1, key2 = key
        key1 = self._key_to_indices(key1)
        key2 = self._key_to_indices(key2)
        indices = self._get_tril_index_from_indices(key1, key2)
        return self.array[indices]


class MaxTanimotoCVMethod(CVMethod):

    def __init__(self, score_mat=None, *args, **kwargs):
        super(MaxTanimotoCVMethod, self).__init__(*args, **kwargs)
        self.score_mat = score_mat
        self.train_target_mol_dict = {}
        self.train_fp_array = None
        self.train_target_fp_inds_dict = {}

    def is_trained(self, target_list):
        if self.score_mat is not None:
            if not self.score_mat.is_loaded():
                return False
            if set(target_list).issubset(self.train_target_mol_dict.keys()):
                return True
        else:
            if not self.train_fp_array:
                return False
            if set(target_list).issubset(
                    self.train_target_fp_inds_dict.keys()):
                return True
        return False

    def train(self, fp_array, mol_to_fp_inds, target_mol_array,
              target_list, mol_list, mask):
        """Train targets by storing mol names for later lookup.

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

        if self.score_mat is not None:
            if not self.score_mat.is_loaded():
                self.score_mat.load()
        else:
            if issparse(fp_array):
                logging.info("Converting from sparse to dense fingerprints.")
                fp_array = fp_array.toarray()
            self.train_fp_array = fp_array

        logging.info("Fitting targets.")
        for i, target_key in enumerate(target_list):
            pos_mol_inds = np.where(target_mol_array[i, :] &
                                    mask[i, :])[0]
            if self.score_mat is not None:
                pos_mols = [mol_list[j] for j in pos_mol_inds]
                self.train_target_mol_dict[target_key] = pos_mols
            else:
                self.train_target_fp_inds_dict[target_key] = [
                    y for x in pos_mol_inds for y in mol_to_fp_inds[x]]
        logging.info("Finished fitting targets.")

    def test(self, fp_array, mol_to_fp_inds, target_mol_array, target_list,
             mol_list, mask):
        """Score test molecules against training targets by max Tanimoto.

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
                          dtype=RESULTS_DTYPE) * self.default_pred

        if self.score_mat is None and issparse(fp_array):
            logging.info("Converting from sparse to dense fingerprints.")
            fp_array = fp_array.toarray()

        for i, target_key in enumerate(target_list):
            logging.debug("Searching {} against molecules ({}/{}).".format(
                target_key.tid, i, target_num))

            # get subset of test data
            test_mol_inds = np.where(mask[i, :])[0]

            if self.score_mat is not None:
                test_mol_names = [mol_list[j] for j in test_mol_inds]
                target_mol_names = self.train_target_mol_dict[target_key]

                # perform test
                tcs = self.score_mat[target_mol_names, test_mol_names]
                scores = np.amax(tcs, axis=0)
            else:
                target_fp_inds = self.train_target_fp_inds_dict[target_key]
                test_mol_fp_start_inds = []
                test_fp_inds = []
                for x in test_mol_inds:
                    mol_fp_inds = mol_to_fp_inds[x]
                    test_mol_fp_start_inds.append(len(test_fp_inds))
                    test_fp_inds.extend(mol_fp_inds)

                # perform test
                tcs = tanimoto(self.train_fp_array[target_fp_inds, :],
                               fp_array[test_fp_inds, :])
                scores = np.amax(tcs, axis=0)
                scores = np.maximum.reduceat(scores, test_mol_fp_start_inds)
            results[i, test_mol_inds] = scores
        return results


class SEASearchCVMethod(CVMethod):

    """Cross-validation method using the Similarity Ensemble Approach.

    Attributes
    ----------
    out_dir : str
        Directory in which to save any output files.
    fit_file : str
        File in which to save background fit.
    library_file : str
        SEA library file in which to store training data for searching.
    train_molecules_file : str
        SEA molecules file storing molecules for training.
    train_targets_file : str
        SEA molecules file storing targets for training.
    test_molecules_file : str
        SEA molecules file storing molecules for testing.
    test_targets_file : str
        SEA molecules file storing targets for testing.
    overwrite : bool
        Overwrite any output files if they already exist.
    """

    fit_file_ext = ".sea"

    def __init__(self, *args, **kwargs):
        self.library_file = None
        self.fit_file = None
        self.train_molecules_file = None
        self.train_targets_file = None
        self.test_molecules_file = None
        self.test_targets_file = None
        super(SEASearchCVMethod, self).__init__(*args, **kwargs)

    @CVMethod.out_dir.setter
    def out_dir(self, out_dir):
        CVMethod.out_dir.fset(self, out_dir)
        self.library_file = os.path.join(self.out_dir, "library.sea")
        self.fit_file = os.path.join(self.out_dir, "library.fit")
        self.train_molecules_file = os.path.join(self.out_dir,
                                                 "train_molecules.csv.bz2")
        self.train_targets_file = os.path.join(self.out_dir,
                                               "train_targets.csv.bz2")
        self.test_molecules_file = os.path.join(self.out_dir,
                                                "test_molecules.csv.bz2")
        self.test_targets_file = os.path.join(self.out_dir,
                                              "test_targets.csv.bz2")

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
                          self.train_targets_file, self.fit_file)

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
                          dtype=RESULTS_DTYPE) * self.default_pred
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

    """Base class for defining classifier methods.

    Attributes
    ----------
    out_dir : str
        Directory in which to save any output files.
    fit_dir : str
        Directory in which to store target fits.
    overwrite : bool
        Overwrite any output files if they already exist.
    """

    dtype = np.float64
    dense_data = False
    fit_file_ext = ".pkl.bz2"
    train_batch_size = None
    test_batch_size = None
    train_sample_negatives = False

    def __init__(self, *args, **kwargs):
        self.fit_dir = None
        super(ClassifierCVMethodBase, self).__init__(*args, **kwargs)

    @CVMethod.out_dir.setter
    def out_dir(self, out_dir):
        CVMethod.out_dir.fset(self, out_dir)
        self.fit_dir = os.path.join(self.out_dir, "target_fits")

    @classmethod
    def create_clf(cls, data=None):
        """Initialize new classifier."""
        raise NotImplementedError

    @classmethod
    def train_clf(cls, self, data, result, batch_size=None):
        """Train classifier with data and result, optionally in batches."""
        raise NotImplementedError

    @classmethod
    def score_clf(cls, clf, data, result):
        """Score trained classifier."""
        raise NotImplementedError

    @classmethod
    def calculate_pred(cls, clf, data):
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
        mask_row : ndarray of bool (n_mols,)
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

        touch_dir(self.fit_dir)
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
        """Score test molecules against training targets using classifier.

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
                          dtype=RESULTS_DTYPE) * self.default_pred

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

    @classmethod
    def train_clf(cls, clf, data, result, batch_size=None):
        """Train classifier with data and result."""
        clf.fit(data, result)

    @classmethod
    def score_clf(cls, clf, data, result):
        """Score trained classifier."""
        return clf.score(data, result)

    @classmethod
    def calculate_pred(cls, clf, data):
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
    kernel = staticmethod(tanimoto)
    random_state = RANDOM_STATE

    @classmethod
    def create_clf(cls, data=None):
        """Create SVM classifier with tanimoto kernel."""
        return svm.SVC(kernel=cls.kernel, random_state=cls.random_state)

    @classmethod
    def calculate_pred(cls, clf, data):
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
    penalty = "l1"
    loss = "squared_hinge"
    tol = 1e-4
    random_state = RANDOM_STATE

    @classmethod
    def create_clf(cls, data=None):
        """Create SVM classifier with linear kernel."""
        return svm.LinearSVC(penalty=cls.penalty, dual=False,
                             class_weight="balanced", loss=cls.loss,
                             tol=cls.tol, random_state=cls.random_state)

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        clf.sparsify()
        return super(LinearSVMCVMethod, self).save_fit_file(target_key, clf)


class RandomForestCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Random Forest."""

    n_estimators = 100
    max_depth = 2
    random_state = RANDOM_STATE

    @classmethod
    def create_clf(cls, data=None):
        """Create Random Forest classifier."""
        return RandomForestClassifier(n_estimators=cls.n_estimators,
                                      max_depth=cls.max_depth,
                                      min_samples_split=2, n_jobs=-1,
                                      random_state=cls.random_state,
                                      class_weight="balanced")


class NaiveBayesCVMethod(SKLearnCVMethodBase):

    """Cross-validation method using a Naive Bayesian Classifier."""

    @classmethod
    def create_clf(cls, data=None):
        if (data is None or (not np.issubdtype(data.dtype, np.floating) and
                             data.max() == 1)):
            return BernoulliNB(alpha=1.0, fit_prior=True)
        else:  # data is of float/count type
            return MultinomialNB(alpha=1.0, fit_prior=True)


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

    dense_data = True
    fit_file_ext = ".pkl"
    hidden_num_units = 512
    input_dropout_rate = .1
    hidden_dropout_rate = .25
    leakiness = .01
    hidden_nonlinearity = leaky_rectify
    output_nonlinearity = softmax
    max_epochs = 1000
    patience = 75
    default_bits = 1024
    max_batch_size = 1000
    min_percent_data_in_batch = .2

    @classmethod
    def create_clf(cls, data=None):
        """Create neural network."""
        try:
            bits = data.shape[1]
        except AttributeError:
            bits = cls.default_bits
        net_params = {"layers": [("input", InputLayer),
                                 ("inputdrop", DropoutLayer),
                                 ("hidden", DenseLayer),
                                 ("hiddendrop", DropoutLayer),
                                 ("output", DenseLayer)],
                      "input_shape": (None, bits),
                      "inputdrop_p": cls.input_dropout_rate,
                      "hidden_num_units": cls.hidden_num_units,
                      "hidden_nonlinearity": cls.hidden_nonlinearity,
                      "hiddendrop_p": cls.hidden_dropout_rate,
                      "output_num_units": 2,
                      "output_nonlinearity": cls.output_nonlinearity,
                      "update_learning_rate": cls.leakiness,
                      "max_epochs": cls.max_epochs,
                      "on_epoch_finished": EarlyStopping(
                          patience=cls.patience)}
        clf = NeuralNet(**net_params)
        if data is not None:
            batch_size = min(cls.max_batch_size,
                             int(cls.min_percent_data_in_batch *
                                 data.shape[0]))
            clf.batch_iterator_train = BalancedClassIterator(
                batch_size=batch_size)
            clf.batch_iterator_test = BalancedClassIterator(
                batch_size=batch_size)

        return clf

    @classmethod
    def train_clf(cls, clf, data, result, batch_size=None):
        """Train neural network with data and result."""
        return clf.fit(data, result.astype(np.int32))

    @classmethod
    def score_clf(cls, clf, data, result):
        """Score trained neural network."""
        return clf.score(data, result)

    @classmethod
    def calculate_pred(cls, clf, data):
        """Compute probabilities of positive for dataset."""
        return clf.predict_proba(data)[:, 1]

    def save_fit_file(self, target_key, clf):
        """Save target fit to file."""
        fit_file = self._fit_file_from_target_key(target_key)
        clf.save_params_to(fit_file)

    def load_fit_file(self, fit_file):
        """Load target fit from file."""
        with smart_open(fit_file, "rb") as f:
            fit = pkl.load(f)
        first_weights = fit['hidden'][0]
        clf = self.create_clf(data=first_weights.T)
        clf.load_params_from(fit_file)
        return clf

    def __del__(self):
        self.target_fits.close()
        del self.target_fits
