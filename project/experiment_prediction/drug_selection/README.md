# Preparing Molecules for Searching

## Create Directory Structure

```bash
mkdir e3fp
mkdir ecfp4
```

## Generate Conformers

*See early discussion on [conformer generation](../../conformer_generation).*

```bash
python $E3FP_PACKAGE/conformer/generate.py -s $E3FP_PROJECT/data/chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2 -o conformers_proto_rms0.5 --pool_multiplier 2 -r 0.5 -C 2
```

## Fingerprint Molecules

*See early discussion on [fingerprinting](../../crossvalidation/sea/README.md).*

To generate ECFP4 fingerprints:

```bash
python $E3FP_PROJECT/scripts/generate_ecfp_fprints.py $E3FP_PROJECT/data/chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2 -o ecfp4/molecules.csv.bz2 -l ecfp4/fp_log.txt
```

To generate E3FP fingerprints using an early optimal parameter set:

```bash
python $E3FP_PROJECT/scripts/generate_e3fp_fprints.py $E3FP_PROJECT/data/chembl20_zinc_in-stock_mw800_unique_proto.smi.bz2 $E3FP_PROJECT/data/early_params.cfg --sdf_dir conformers_proto_rms0.5 -o e3fp/molecules.csv.bz2 -l e3fp/fp_log.txt
```
