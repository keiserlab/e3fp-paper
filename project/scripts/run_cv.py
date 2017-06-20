"""Run k-fold cross-validation.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import logging
import argparse

from python_utilities.parallel import Parallelizer, ALL_PARALLEL_MODES
from python_utilities.scripting import setup_logging
from e3fp_paper.crossvalidation.util import InputProcessor
from e3fp_paper.crossvalidation.run import KFoldCrossValidator, \
                                           MoleculeSplitter, \
                                           ByTargetMoleculeSplitter
from e3fp_paper.crossvalidation.methods import LinearSVMCVMethod, \
                                               RandomForestCVMethod, \
                                               NaiveBayesCVMethod, \
                                               NeuralNetCVMethod, \
                                               SEASearchCVMethod, \
                                               MaxTanimotoCVMethod,\
                                               RandomCVMethod, \
                                               ScoreMatrix


CV_METHODS = {'sea': SEASearchCVMethod, 'nb': NaiveBayesCVMethod,
              'rf': RandomForestCVMethod, 'linsvm': LinearSVMCVMethod,
              'nn': NeuralNetCVMethod, 'rand': RandomCVMethod,
              'maxtc': MaxTanimotoCVMethod}
SPLITTERS = {'molecule': MoleculeSplitter, 'target': ByTargetMoleculeSplitter}
AFFINITY_CHOICES = [0, 10, 100, 1000, 10000]
PROCESS_CHOICES = ['union', 'mean', 'mean-boltzmann', 'first']
K = 5


def main(molecules_file="molecules.csv.bz2", targets_file="targets.csv.bz2",
         k=5, method='sea', tc_files=None, auc_type='sum', process_inputs=None,
         split_by='target', reduce_negatives=False, min_mols=50,
         affinity=10000, out_dir="./", overwrite=False, log=None,
         num_proc=None, parallel_mode=None, verbose=False):
    setup_logging(log, verbose=verbose)
    if num_proc is None:
        num_proc = k + 1
    parallelizer = Parallelizer(parallel_mode=parallel_mode, num_proc=num_proc)

    cv_class = CV_METHODS[method]
    if cv_class is MaxTanimotoCVMethod and tc_files is not None:
        score_matrix = ScoreMatrix(*tc_files)
        cv_class = cv_class(score_matrix)
    splitter_class = SPLITTERS[split_by]
    if isinstance(splitter_class, MoleculeSplitter):
        splitter = splitter_class(k=k)
    else:
        splitter = splitter_class(k=k, reduce_negatives=reduce_negatives)

    if process_inputs is not None:
        processor = InputProcessor(mode=process_inputs)
    else:
        processor = None

    kfold_cv = KFoldCrossValidator(k=5, parallelizer=parallelizer,
                                   splitter=splitter,
                                   input_processor=processor,
                                   cv_method=cv_class,
                                   return_auc_type=auc_type, out_dir=out_dir,
                                   overwrite=overwrite)
    auc = kfold_cv.run(molecules_file, targets_file, min_mols=min_mols,
                       affinity=affinity)
    logging.info("CV Mean AUC: {:.4f}".format(auc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """Run k-fold cross-validation.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('molecules_file', type=str,
                        help="""Path to SEA-format molecules file.""")
    parser.add_argument('targets_file', type=str,
                        help="""Path to SEA-format targets file.""")
    parser.add_argument('-k', type=int, default=K,
                        help="""Number of folds into which to split data.""")
    parser.add_argument('--method', type=str, choices=CV_METHODS.keys(),
                        default='sea',
                        help="""Type of classifier to use for training and
                             testing.""")
    parser.add_argument('--tc_files', type=str, nargs=2, default=None,
                        metavar=('memmap_file', 'mol_names_file'),
                        help="""Files necessary to build pairwise Tanimoto
                             matrix, to be used only with 'maxtc' method.
                             memmap_file is a NumPy memory-mapped file of
                             with a flat array of Tanimoto Coefficients (TC)
                             from lower triangle of the pairwise TC matrix,
                             while mol_names_file is a file with names of the
                             mols in the matrix.""")
    parser.add_argument('--auc_type', type=str,
                        choices=['prc', 'roc', 'sum'], default='sum',
                        help="""Type of AUC to use for objective function.""")
    parser.add_argument('--process_inputs', type=str, choices=PROCESS_CHOICES,
                        default=None,
                        help="""Process all fingerprints for a molecule
                             before cross-validation.""")
    parser.add_argument('--split_by', type=str, choices=SPLITTERS.keys(),
                        default='target',
                        help="""How to split molecules into folds. 'target'
                             ensures that each target has 1/k positive mols in
                             its test set and (k-1)/k in its training set.
                             'molecule' randomly splits molecules into k
                             sets.""")
    parser.add_argument('--reduce_negatives', action='store_true',
                        help="""When folds are balanced by targets with
                             '--split_by' of 'target', remove all mols from
                             negatives that are not positives for any target.
                             When negatives greatly outnumber positives, this
                             has little effect on result but increases
                             computational efficiency.""")
    parser.add_argument('--min_mols', type=int, default=50,
                        help="""Minimum number of known binders to a target
                             in order to consider that target.""")
    parser.add_argument('--affinity', type=int,
                        choices=AFFINITY_CHOICES, default=10000,
                        help="""Minimum affinity level of binders to
                             consider.""")
    parser.add_argument('-o', '--out_dir', type=str, default="./",
                        help="""Output directory.""")
    parser.add_argument('-O', '--overwrite', action="store_true",
                        help="""Overwrite existing file(s).""")
    parser.add_argument('-l', '--log', type=str, default=None,
                        help="Log filename.")
    parser.add_argument('-p', '--num_proc', type=int, default=None,
                        help="""Set number of processors to use. Defaults to
                             k + 1""")
    parser.add_argument('--parallel_mode', type=str, default=None,
                        choices=list(ALL_PARALLEL_MODES),
                        help="""Set parallelization mode to use.""")
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="Run with extra verbosity.")
    params = parser.parse_args()

    kwargs = dict(params._get_kwargs())
    main(**kwargs)
