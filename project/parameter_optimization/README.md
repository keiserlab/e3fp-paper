# Parameter Optimization with Spearmint

Each of the following directories contains a config
directory. The files within this directory are organized
according to Spearmint's requirements.

## Running the parameter optimization

1. `cd` into the `spearmint` subdirectory of the spearmint
installation. See the [spearmint readme](https://github.com/JasperSnoek/spearmint/blob/master/spearmint/#running-the-automated-code-spearmint)
for more details.
2. Run first optimization with `python main.py --driver=local --method=GPEIperSecChooser --method-args=noiseless=1 $E3FP_PROJECT/parameter_optimization/1_chembl20_opt/config/config.pb`
3. Run first optimization with `python main.py --driver=local --method=GPEIperSecChooser --method-args=noiseless=1 $E3FP_PROJECT/parameter_optimization/2_chembl20_opt/config/config.pb`

* Note: For the actual paper, a random 100,000 molecules
from ChEMBL20 and all targets to which at least 50 of
these molecules bound were used for computational
efficiency. It's highly recommended to use a parallelization
method when running parameter optimization.
