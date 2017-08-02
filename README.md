# Extended 3-Dimensional FingerPrint (E3FP) Paper Materials

[E3FP](https://github.com/keiserlab/e3fp) is a computational method for
generating 3D molecular fingerprints. This repository serves as an application
of E3FP and contains a Python 2.7.x-compatible library and all scripts
necessary to reproduce the analyses and figures in the E3FP
paper<sup>[1](#axen2017)</sup>.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
    + [Required](#required)
        - [Required for Library](#requiredlib)
        - [Required for Project](#requiredproj)
    + [Optional](#optional)
- [Setup and Installation](#setup)
- [Usage and Overview](#usage)
- [References](#references)

<a name="overview"></a>
## Overview

This repository is divided into two sections:

- [`e3fp_paper`](e3fp_paper) is a Python library containing various classes
  and methods used in the paper analysis. Specifically, it contains code for
  interfacing with SeaWare, an implementation of the Similarity Ensemble
  Approach (SEA), loading and saving SEA-compatible filetypes, running
  *k*-fold cross-validation, and plotting the results.
- [`project`](project) contains all scripts necessary to run the analyses in
  the paper. While instructions are provided, please see the E3FP
  paper<sup>[1](#axen2017)</sup> (preprint available) for detailed
  explanations.

<a name="dependencies"></a>
## Dependencies

`e3fp_paper` is compatible with Python 2.7.x. It additionally has the following
dependencies:

<a name="required"></a>
### Required

<a name="requiredlib"></a>
#### Required for Library

The following packages and their dependencies must be installed:

- [e3fp](https://github.com/keiserlab/e3fp)
- SeaWare
- [RDKit](http://www.rdkit.org)
- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org)
- [Pandas](http://pandas.pydata.org)
- [Matplotlib](http://matplotlib.org)
- [scikit-learn](http://scikit-learn.org)
- [nolearn](https://github.com/dnouri/nolearn)
- [python_utilities](https://github.com/sdaxen/python_utilities)

<a name="requiredproj"></a>
#### Required for Project

In addition to the above packages, these must be installed to run the project
scripts.

- [Spearmint](https://github.com/JasperSnoek/spearmint)
- [Seaborn](https://seaborn.pydata.org)
- [NetworkX](https://networkx.github.io)
- [PyGraphviz](https://pygraphviz.github.io)
- [PyMOL](https://www.pymol.org)

<a name="optional"></a>
### Optional

Some computationally expensive analyses have built-in acceleration with
[python_utilities](https://github.com/sdaxen/python_utilities) that activates
when one of the following packages is installed:

- [mpi4py](http://mpi4py.scipy.org)
- [futures](https://pypi.python.org/pypi/futures)

<a name="setup"></a>
## Setup and Installation

Before installing, you must manually install [RDKit](http://www.rdkit.org),
SeaWare, [Spearmint](https://github.com/JasperSnoek/spearmint), and
[PyMOL](https://www.pymol.org). Additionally you will need `pip` and
`setuptools`.

### Clone the Repository
0. Install any of the optional dependencies above.
1. Download this repository to your machine.
    - Clone this repository to your machine with
      `git clone https://github.com/keiserlab/e3fp-paper.git`.
    - OR download an archive by navigating to
      [https://github.com/keiserlab/e3fp-paper](https://github.com/keiserlab/e3fp-paper)
      and clicking "Download ZIP". Extract the archive.
2. Install with
    ```bash
    cd e3fp-paper
    pip install .
    ```

<a name="usage"></a>
## Usage and Overview

To use the Python library in a python script, enter: 
```python
import e3fp_paper
```
See [`pipeline.py`](e3fp_paper/pipeline.py) for methods for interfacing E3FP's
[pipeline](https://github.com/keiserlab/e3fp/blob/master/e3fp/pipeline.py)
with the specific filetypes used in the paper library.

See the provided [scripts](project/scripts) for applications of E3FP and of
the `e3fp_paper` library.

<a name="references"></a>
## References
<a name="axen2017"></a>
1. Axen SD, Huang XP, Caceres EL, Gendelev L, Roth BL, Keiser MJ. A Simple
   Representation Of Three-Dimensional Molecular Structure.
   *J. Med. Chem.* (2017).
   doi: [10.1021/acs.jmedchem.7b00696](http://dx.doi.org/10.1021/acs.jmedchem.7b00696). \
   <a href="http://f1000.com/prime/727824514?bd=1" target="_blank"><img src="http://cdn.f1000.com.s3.amazonaws.com/images/badges/badgef1000.gif" alt="Access the recommendation on F1000Prime" id="bg" /></a>
