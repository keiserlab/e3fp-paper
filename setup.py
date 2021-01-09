try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

requirements = [
    'e3fp>=1.1',
    'sdaxen_python_utilities>=0.1.4',
    'scipy>=0.18.0',
    'numpy>=1.11.3',
    'matplotlib>=2.0.0',
    'networkx>=1.11',
    'nolearn>=0.6.0',
    'pandas>=0.18.1',
    'pygraphviz>=1.3.1',
    'scikit-learn>=0.18.0',
    'seaborn>=0.7.1'
]

classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2 :: Only',
               'Programming Language :: Python :: 2.7',
               'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
               'Operating System :: OS Independent',
               'Development Status :: 4 - Beta',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'Topic :: Scientific/Engineering :: Chemistry',
               'Topic :: Software Development :: Libraries :: Python Modules'
               ]

setup(
    name='e3fp_paper',
    packages=['e3fp_paper', 'e3fp_paper.crossvalidation',
              'e3fp_paper.plotting', 'e3fp_paper.sea_utils'],
    version='1.1',
    description='E3FP paper library and repo',
    keywords='e3fp 3d molecule fingerprint conformer paper',
    author='Seth Axen',
    author_email='seth.axen@gmail.com',
    license='LGPLv3',
    url='https://github.com/keiserlab/e3fp-paper',
    classifiers=classifiers,
    download_url='https://github.com/keiserlab/e3fp-paper/tarball/1.1',
    install_requires=requirements,
    include_package_data=True
)
