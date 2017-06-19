"""Make table summarizing cross-validation results from E3FP/ECFP variants.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import re
import os
import glob

import numpy as np
import pandas as pd

CV_BASEDIR = os.path.join(os.environ['E3FP_PROJECT'], 'crossvalidation',
                          'sea')
E3FP_NOSTEREO_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR,
                                                   "e3fp-nostereo*"))
E3FP_REPEAT_DIRS = [x for x in glob.glob(os.path.join(CV_BASEDIR, "e3fp*"))
                    if x not in E3FP_NOSTEREO_REPEAT_DIRS]
E2FP_STEREO_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR, "e2fp-stereo*"))
E2FP_REPEAT_DIRS = [x for x in glob.glob(os.path.join(CV_BASEDIR, "e2fp*"))
                    if x not in E2FP_STEREO_REPEAT_DIRS]
ECFP_CHIRAL_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR, "ecfp4-chiral*"))
ECFP_REPEAT_DIRS = [x for x in glob.glob(os.path.join(CV_BASEDIR, "ecfp4*"))
                    if x not in ECFP_CHIRAL_REPEAT_DIRS]
E3FP_RDKIT_REPEAT_DIRS = glob.glob(os.path.join(CV_BASEDIR, "e3fp-rdkit*"))
CVSTATS_FILE_NAME = "table_1.txt"


def stats_from_cv_dirs(cv_dirs):
    aurocs = []
    auprcs = []
    target_aurocs = []
    target_auprcs = []
    target_perc_pos = []
    for cv_dir in cv_dirs:
        log_file = os.path.join(cv_dir, "log.txt")
        with open(log_file, "rU") as f:
            for line in f:
                try:
                    m = re.search(('Target.*AUROC of (0\.\d+).*AUPRC of '
                                   '(0\.\d+)\.\s\((0\.\d+)'), line)
                    auroc, auprc, perc_pos = [float(m.group(i))
                                              for i in range(1, 4)]
                    target_aurocs.append(auroc)
                    target_auprcs.append(auprc)
                    target_perc_pos.append(perc_pos)
                    continue
                except AttributeError:
                    pass
                try:
                    m = re.search(
                        'Fold.*AUROC of (0\.\d+).*AUPRC of (0\.\d+)', line)
                    auroc, auprc = float(m.group(1)), float(m.group(2))
                    aurocs.append(auroc)
                    auprcs.append(auprc)
                except AttributeError:
                    pass
    return ((np.mean(auprcs), np.std(auprcs)),
            (np.mean(aurocs), np.std(aurocs)),
            (np.mean(target_auprcs), np.std(target_auprcs)),
            (np.mean(target_aurocs), np.std(target_aurocs)),
            (np.mean(target_perc_pos), np.std(target_perc_pos)))


if __name__ == "__main__":
    names = ["ECFP4", "ECFP4-Chiral", "E2FP", "E2FP-Stereo", "E3FP-NoStereo",
             "E3FP", "E3FP-RDKit"]
    dirs_list = [ECFP_REPEAT_DIRS, ECFP_CHIRAL_REPEAT_DIRS,
                 E2FP_REPEAT_DIRS, E2FP_STEREO_REPEAT_DIRS,
                 E3FP_NOSTEREO_REPEAT_DIRS,
                 E3FP_REPEAT_DIRS, E3FP_RDKIT_REPEAT_DIRS]
    stats = []
    for dirs in dirs_list:
        stats.append(stats_from_cv_dirs(dirs))

    stats_strs = [["{:.4f} +/- {:.4f}".format(*pair) for pair in row]
                  for row in stats]
    df = pd.DataFrame(stats_strs, columns=[
        "Mean Fold AUPRC", "Mean Fold AUROC", "Mean Target AUPRC",
        "Mean Target AUROC", "Positive Data Pairs"])
    df['Name'] = names
    df.set_index('Name', inplace=True)
    with open(CVSTATS_FILE_NAME, "w") as f:
        df.to_csv(f, sep='\t')
