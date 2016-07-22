"""Utilities for searching molecules against a SEA library.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import os
import logging
import shelve

from zc.lockfile import LockError

from seacore.run.core import SEASetCore
from seacore.util.library import TargetKey


class SEASetSearcher(object):

    """Wrapper for easily searching molecule sets against a SEA library."""

    def __init__(self, reference, log=False, mol_db=None, target_db=None,
                 wipe=True):
        """Constructor.

        Parameters
        ----------
        reference : str
            Path to SEA library file.
        log : bool, optional
            Create log messages.
        mol_db : str or None, optional
            If path is provided, create a database of dynamically updated
            molecule results.
        target_db : str or None, optional
            If path is provided, create a database of dynamically updated
            target results.
        wipe : bool, optional
            Clear search results (including database) when opening library.
        """
        self.log = log
        self.wipe = wipe
        self.core = SEASetCore()
        if mol_db is not None:
            self.set_results_dict = shelve.open(mol_db, writeback=True)
        else:
            self.set_results_dict = {}
        if target_db is not None:
            self.target_results_dict = shelve.open(target_db, writeback=True)
        else:
            self.target_results_dict = {}
        self.target_id_key_map = {}
        self.open(reference)

    def open(self, reference):
        """Open SEA library file. If already open, close and then reopen.

        Parameters
        ----------
        reference : str
            Path to library file.
        """
        if self.wipe:
            self.clear_results()
        if self.core.library is not None:
            logging.error(
                "Reference library for search is already defined. Closing.",
                exc_info=True)
            self.close()
        if self.log:
            logging.info("Loading library for searching.")
        self.core.load_reference(reference)
        if self.log:
            logging.info("Library loaded.")

    def search(self, mol_name, fp_list, rerun=False):
        """Search set of fingerprints for one molecule against library.

        Parameters
        ----------
        mol_name : str
            Name of molecule
        fp_list : list of tuple
            List of native tuples, each tuple containing native string and
            fingerprint name.
        rerun : bool, optional
            If True, rerun search if result already exists.

        Returns
        -------
        mol_results : dict
            Dict of results. Keys are ``TargetKey``, values are a tuple of
            e-value and maximum tanimoto coefficient.
        """
        if not rerun and mol_name in self.set_results_dict:
            return self.set_results_dict[mol_name]
        try:
            results = self.core.run(fp_list)
        except AttributeError:
            logging.error(
                "Reference library for search has not been defined.",
                exc_info=True)
            return {}

        logging.debug("SEA Search Results: \n" + repr(results.results))
        # self.set_results_dict[mol_name] = {
        #     target_id: (evalue, max_tc)
        #     for _, target_id, hit_affinity, evalue, max_tc
        #     in results.results}
        for r in results.results:
            _, target_id, hit_affinity, evalue, max_tc = r
            target_key = TargetKey(target_id, hit_affinity)
            self.set_results_dict.setdefault(
                mol_name, {})[target_key] = (evalue, max_tc)
            self.target_results_dict.setdefault(target_key,
                                                set([])).add(mol_name)
            self.target_id_key_map.setdefault(target_id, set()).add(target_key)

        try:
            self.set_results_dict.sync()
        except AttributeError:
            pass
        try:
            self.target_results_dict.sync()
        except AttributeError:
            pass
        return self.set_results_dict.get(mol_name, {})

    def batch_search(self, molecules_lists_dict):
        """Search multiple fingerprint sets against SEA library.

        Parameters
        ----------
        molecules_lists_dict : dict
            Dict matching molecule name to list of native_tuples.

        Returns
        -------
        success : bool
            True if batch search was successful, False if not.
        """
        if self.core.library is None:
            logging.error(
                "Reference library for search has not been defined.",
                exc_info=True)
            return False
        logging.info("Searching sets against library.")
        search_iter = (self.search(k, v)
                       for k, v in molecules_lists_dict.iteritems())
        i = 0
        for result in search_iter:
            if (i + 1) % 100 == 0:
                print("Searched {:d} molecules.".format(i + 1))
            i += 1
        if self.log:
            logging.info(
                "Searched {:d} molecule sets against library.".format(i + 1))
        return True

    def mol_result(self, mol_name):
        """Get results for molecule.

        Parameters
        ----------
        mol_name : str
            Name of molecule.

        Returns
        -------
        mol_results : dict
            Dict of results. Keys are ``TargetKey``, values are a tuple of
            e-value and maximum tanimoto coefficient.
        """
        return self.set_results_dict.get(mol_name, {})

    def target_result(self, target_id, affinity=None):
        """Get results for target at any affinity levels.

        To retrieve stats for molecule/target pairs, follow with
        'mol_target_result'.

        Parameters
        ----------
        target_id : str
            Target id.
        affinity : int or iterable
            One or more specific affinity levels at which to retrieve results.
            If not provided, all are returned.

        Returns
        -------
        target_results : set
            Set of molecule names which hit to target.
        """
        target_keys = self._get_target_keys(target_id, affinity=affinity)

        return set.union(*[self.target_results_dict.get(target_key, set())
                           for target_key in target_keys])

    def mol_target_result(self, mol_name, target_id, affinity=None):
        """Get results for mol/target pair at any/all affinities.

        Parameters
        ----------
        mol_name : str
            Name of molecule.
        target_id : str
            Target id.

        Returns
        -------
        mol_target_results : dict or None
            Dict matching target keys at provided affinities to the tuple of
            e-value and max tanimoto coefficient if a significant hit
            was found between mol and target at that affinity. Otherwise,
            None is returned.
        """
        target_keys = self._get_target_keys(target_id, affinity=affinity)
        mol_results = self.set_results_dict.get(mol_name, {})
        results = {x: mol_results[x] for x in target_keys if x in mol_results}
        if len(results) == 0:
            return None
        else:
            return results

    def _get_target_keys(self, target_id, affinity=None):
        """Get target keys corresponding to id at specific affinities.

        Parameters
        ----------
        target_id : str
            Target id
        affinity : iterable or int or None, optional
            An affinity or iterable of affinity levels. If None, all affinity
            levels are considered.
        """
        target_keys = set()
        if affinity is None:
            target_keys.update(self.target_id_key_map.get(target_id, set()))
        elif isinstance(affinity, (int, long, str)):  # single affinity
            return {TargetKey(target_id, str(affinity))}
        else:  # iterable of affinities
            affinity = set(affinity)
            target_keys.update([
                x for x in self.target_id_key_map.get(target_id, set())
                if int(x.group) in affinity])
        return target_keys

    def close(self):
        """Close library file."""
        if self.log:
            logging.info("Closing library.")
        self.core.close()

    def clear_results(self):
        """Clear all results."""
        if self.log:
            logging.info("Clearing results from searcher.")
        self.set_results_dict.clear()
        self.target_results_dict.clear()
        self.target_id_key_map.clear()

    def __del__(self):
        self.close()


def sea_set_search(library_file, molecules_lists_dict, log=True):
    """Get ``SEASetSearcher`` for molecule lists against a SEA library.

    Parameters
    ----------
    library_file : str
        Path to SEA library.
    molecules_lists_dict : dict
        Dict matching molecule name to list of native tuples.
    log : bool, optional
        Create log messages.

    Returns
    -------
    searcher : SEASetSearcher
        Instance of searcher used, containing results of batch search, for
        convenience so results can be iterated or requested in any form.
    """
    try:
        searcher = SEASetSearcher(library_file, log=log)
    except LockError:
        # SEA locks the library upon opening so it may only be opened once.
        # However, sometimes the lock file is not deleted, so we ignore it.
        os.remove("{!s}.lock".format(library_file))
        searcher = SEASetSearcher(library_file, log=log)

    searcher.batch_search(molecules_lists_dict)
    searcher.close()
    return searcher
