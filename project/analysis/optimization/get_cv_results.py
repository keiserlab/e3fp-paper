"""Read cross-validation log files and save runtime and results to file.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import sys
import glob
import os
from datetime import datetime
import ast
import logging

import numpy as np
import pandas as pd
from python_utilities.scripting import setup_logging


EXPECTED_CV_NUM = 5


def get_time_from_line(line):
    time_str = line.split('|')[0].split(','[0])[0]
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return time


def delta_time_to_hours(time_delta):
    return time_delta.days * 24. + time_delta.seconds / 60. / 60.


def get_results_from_cv_dir(cv_dir):
    total_runtime = 0
    num_proc = None
    params = {}
    fold_tc_cutoffs = []
    fold_auprcs = []
    fold_aurocs = []

    log_file = os.path.join(cv_dir, "log.txt")
    fit_files = glob.glob(os.path.join(cv_dir, "*/*fit"))
    if not os.path.isfile(log_file):
        raise ValueError("{} does not have a log file.".format(
            os.path.basename(cv_dir)))

    with open(log_file, "r") as f:
        line = next(f)
        start_time = get_time_from_line(line)
        while "Parallelizer initialized" not in line:
            line = next(f)
        num_proc = int(line.split(" ")[-2])

        line = next(f)
        params_str = line.split("Params: ")[1]
        params = dict(ast.literal_eval(params_str))

        count_fprint_time = True
        while True:
            line = next(f)
            if "Generating fingerprints from conformers" in line:
                while "Generating fingerprints for" not in line:
                    line = next(f)
                break
            elif "Molecules file already exists" in line:
                count_fprint_time = False
                break

        if count_fprint_time:
            fprint_start_time = get_time_from_line(line)
            last_line = line
            line = next(f)
            while "Generat" in line and "fingerprints for" in line:
                last_line = line
                line = next(f)
            fprint_end_time = get_time_from_line(last_line)
            fprint_runtime = delta_time_to_hours(
                fprint_end_time - fprint_start_time) * max(1, num_proc - 1)
            io_start_time = fprint_end_time
        else:
            fprint_runtime = 0
            io_start_time = get_time_from_line(line)

        while True:
            line = next(f)
            if ("Running fold validation" in line or
                    "Building library" in line):
                break

        io_end_time = get_time_from_line(line)
        io_runtime = delta_time_to_hours(io_end_time - io_start_time)

        cv_start_time = io_end_time
        fold_cv_runtimes = []
        while "CV Mean" not in line:
            try:
                line = next(f)
            except StopIteration:
                break
            if ("|Mean AUC:" in line or ("Fold" in line and
                                         "produced an AUROC" in line)):
                fold_end_time = get_time_from_line(line)
                fold_cv_runtimes.append(delta_time_to_hours(fold_end_time -
                                                            cv_start_time))

                if "Fold" in line:
                    fold = int(line.split("Fold ")[1].split()[0])
                    fold_auroc = float(line.split("AUROC of ")[1].split()[0])
                    fold_auprc = float(
                        line.split("AUPRC of ")[1].split()[0][:-1])
                else:  # old log file format, assume AUC is AUPRC
                    fold = int(line.split("(")[1].split()[0])
                    fold_auprc = float(line.split("AUC: ")[1].split()[0])
                    fold_auroc = np.nan
                fold_auprcs.append((fold, fold_auprc))
                fold_aurocs.append((fold, fold_auroc))

        end_time = get_time_from_line(line)
        cv_runtime = sum(fold_cv_runtimes)
        total_runtime = fprint_runtime + io_runtime + cv_runtime
        real_runtime = delta_time_to_hours(end_time - start_time)
        assert(len(fold_auprcs) == EXPECTED_CV_NUM)

    fold_aurocs = zip(*sorted(fold_aurocs))[1]
    fold_auprcs = zip(*sorted(fold_auprcs))[1]

    if len(fit_files) > 0:
        for fit_file in sorted(fit_files):
            with open(fit_file, "r") as f:
                for line in f:
                    if "TANI" in line:
                        tc = float(line.split()[-1])
                        fold_tc_cutoffs.append(tc)
    else:
        fold_tc_cutoffs = [np.nan for x in fold_auprcs]

    return (start_time, total_runtime, real_runtime, num_proc, fold_auprcs,
            fold_aurocs, fold_tc_cutoffs, params)


if __name__ == "__main__":
    usage = "python get_cv_results.py <cv_run_dir> <out_results_table>"
    try:
        run_dir, out_file = sys.argv[1:3]
    except (IndexError, ValueError):
        sys.exit(usage)

    setup_logging(verbose=True)

    cv_dirs = glob.glob(os.path.join(run_dir, "e3fp*/"))

    results = {"Start Time": [], "Runtime": [], "Real Runtime": [],
               "Processor Number": [], "Mean AUPRC": [], "Std AUPRC": [],
               "Mean AUROC": [], "Std AUROC": [], "First": [],
               "Radius Multiplier": [], "Level": [], "Conformers": [],
               "Bits": [], "Fold": [], "Fold AUPRC": [], "Fold AUROC": [],
               "Fold TC Cutoff": []}
    for i, cv_dir in enumerate(cv_dirs):
        logging.info("Reading cv_dir {} ({:d}/{:d})".format(cv_dir, i + 1,
                                                            len(cv_dirs)))
        try:
            (start_time, runtime, real_runtime, num_proc, fold_auprcs,
             fold_aurocs, fold_tc_cutoffs, params) = get_results_from_cv_dir(
                cv_dir)
        except Exception as e:
            logging.error("Error reading cv_dir {}:".format(cv_dir))
            continue

        for i, (auprc, auroc, tc) in enumerate(zip(fold_auprcs, fold_aurocs,
                                                   fold_tc_cutoffs)):
            results["Runtime"].append(runtime)
            results["Real Runtime"].append(real_runtime)
            results["Start Time"].append(start_time)
            results["Processor Number"].append(num_proc)
            results["Mean AUPRC"].append(np.mean(fold_auprcs))
            results["Std AUPRC"].append(np.std(fold_auprcs))
            results["Mean AUROC"].append(np.mean(fold_aurocs))
            results["Std AUROC"].append(np.std(fold_aurocs))
            results["First"].append(params.get("first", np.nan))
            results["Radius Multiplier"].append(params.get("radius_multiplier",
                                                           np.nan))
            results["Level"].append(params.get("level", np.nan))
            results["Conformers"].append(params.get("conformers", np.nan))
            results["Bits"].append(params.get("bits", np.nan))
            results["Fold"].append(i + 1)
            results["Fold AUPRC"].append(auprc)
            results["Fold AUROC"].append(auroc)
            results["Fold TC Cutoff"].append(tc)

    df = pd.DataFrame.from_dict(results)
    df["AUCSUM"] = df["Mean AUPRC"] + df["Mean AUROC"]
    df.sort_values(by=["Start Time", "Fold"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(out_file, sep="\t")
