# TabMon-Bench

**A controlled benchmark for evaluating when label-free model monitoring can
be trusted under tabular distribution shifts.**

TabMon-Bench treats monitor reliability—not only predictive robustness or
shift detectability—as the object of evaluation. Target labels are withheld
from every monitor and retained exclusively by an offline oracle. The
benchmark separately evaluates performance estimation, shift attribution,
and sequential detection because these tasks have different estimands and
failure modes.

This repository contains the complete experiment, audit, and analysis source
for the Schema 13 research artifact. Generated datasets, fitted models, stream
caches, and result archives are intentionally excluded from version control.

## Research questions

TabMon-Bench asks four questions:

1. How accurately can a monitor estimate predictive performance without
   target labels?
2. Can a monitor recover the feature group manipulated by a controlled shift?
3. Can it raise a calibrated sequential alarm for its declared event target?
4. Can observable diagnostics identify when the monitor itself is likely to
   fail?

The protocol explicitly distinguishes visible drift in \(P(X)\), degradation
of predictive risk, and monitor reliability. A change in one does not imply a
change in the others.

## Experimental scope

### Binary controlled benchmark

The final binary grid contains:

- 5 public tabular datasets;
- 4 calibrated predictors: logistic regression, random forest, XGBoost, and a
  multilayer perceptron;
- 10 random seeds (42--51);
- 31 stream configurations: one deduplicated null plus five shift families,
  three severities, and abrupt/gradual temporal modes;
- 10 batches of 1,000 observations per stream; and
- 6,200 paired predictor streams.

The controlled shift families are covariate, correlated-covariate, concept,
pipeline-corruption, and support-violation shifts. The generator records the
manipulated feature set as *intervention ground truth*; this is not claimed to
be real-world causal ground truth.

### Multiclass extension

Schema 13 also contains 1,240 Covertype predictor streams over the same four
model families, ten seeds, and 31 configurations. Because this extension has
one dataset, its uncertainty intervals support within-dataset replication but
not cross-dataset generalization.

### Methods and declared endpoints

Methods are compared only when they share an endpoint.

| Method | Declared endpoint | Primary role |
|---|---|---|
| AC | classification error | performance estimation |
| DOC | classification-error change | performance estimation |
| ATC | classification error | performance estimation |
| COT | classification error | performance estimation |
| COTT | classification error | performance estimation |
| Calibrated Confidence | excess log loss | risk estimation and risk alarm |
| SHD | selected high-error prevalence | sequential harmful-shift detection |
| XPE | transported log-loss change and feature allocation | diagnosis and attribution |
| DriftSHAP-style monitor | observable feature drift weighted by global importance | drift detection and intervention recovery |

AC, DOC, ATC, COT, and COTT are evaluated on the same cached streams and the
same 0--1 classification-error endpoint. Their estimates are never compared
directly with log loss. SHD and XPE retain their published estimands and are
reported separately.

The scalable XPE backend was checked against the authors' reference
implementation at commit `bac9ef37d8409b5eebf31b363d14864c763375c5` before
being admitted to the final run. The implementation records its repository,
commit, backend, transport-conservation checks, and fidelity thresholds in the
generated audit manifest.

## Audited Schema 13 results

The final analysis used 5,000 dataset--seed crossed-cluster bootstrap
replicates for the binary task and seed-cluster replicates for the one-dataset
multiclass extension. It produced 140 method summaries, 280 paired
comparisons, 98 separate-endpoint summaries, and 90 XPE summaries. All final
input and output audits passed.

Overall mean absolute classification-error estimation error was:

| Task | Method | MAE | 95% cluster CI |
|---|---|---:|---:|
| Binary | COTT | 0.0567 | [0.0464, 0.0719] |
| Binary | COT | 0.0587 | [0.0485, 0.0700] |
| Binary | DOC | 0.0748 | [0.0537, 0.1046] |
| Binary | AC | 0.0748 | [0.0536, 0.1037] |
| Binary | ATC | 0.0763 | [0.0489, 0.1077] |
| Multiclass | COT | 0.1001 | [0.0996, 0.1007] |
| Multiclass | COTT | 0.1085 | [0.1080, 0.1090] |
| Multiclass | DOC | 0.1437 | [0.1432, 0.1442] |
| Multiclass | AC | 0.1457 | [0.1452, 0.1463] |
| Multiclass | ATC | 0.1502 | [0.1498, 0.1506] |

These aggregate values are not a universal ranking: paired and mixed-effects
analyses show substantial dependence on the dataset, predictor, and shift
family.

SHD found a feasible source selector in 3/20 binary dataset--model cells and
3/4 multiclass cells. It detected 0/62 binary and 0/133 multiclass event
streams among applicable cells. A cached binary sensitivity analysis over
`alpha` in `{0.01, 0.05, 0.10}` and `epsilon` in `{0, 0.02, 0.05}` produced no
alarms. These are applicability and power findings under the benchmark's
finite streams, not evidence against SHD's conditional theoretical result.

XPE was evaluated on its predeclared 1,300-stream binary and 260-stream
multiclass subsets. It retained useful directional and attribution information
in several regimes, but its transported loss change was not a calibrated
target-risk estimate on this benchmark. The multiclass result remains a
single-dataset extension.

The earlier controlled analysis also identifies three recurring reliability
boundaries: label-only concept changes can be harmful while observable
feature signals remain unchanged; pipeline corruption can invert a learned
confidence--loss surrogate; and failure diagnostics transfer poorly to an
unseen dataset. These findings motivate reporting failure maps rather than a
single average score.

## Leakage controls and audit policy

The deployment-time monitor API excludes:

- target labels;
- oracle risk or loss;
- monitor-failure labels; and
- intervention ground truth.

An explicit feature allowlist protects the meta-monitor from target-derived
diagnostics. Oracle quantities are constructed in a separate evaluation
layer. Every major stage writes a status file, manifest, row-count checks, and
SHA-256 input provenance; audit scripts fail closed on incomplete grids,
duplicate identifiers, endpoint mixing, or an oracle-visible monitor API.

## Repository layout

```text
data/          Local raw/processed data (ignored by Git)
experiments/   Runners, rescorers, audits, and statistical analyses
results/       Generated models and results (ignored by Git)
src/           Models, shifts, monitors, cache, and evaluation code
tests/         Unit and integration tests
```

Historical notebook instructions and schema-transition notes are maintained
internally and are not part of the public source release. Public experiment
entry points and their required arguments are available through the command
line help of the scripts in `experiments/`.

## Installation

Python 3.12 is the recorded execution environment. Core numerical and model
versions are pinned in [`requirements.txt`](requirements.txt), including
scikit-learn 1.6.1 and XGBoost 3.2.0; data-access and plotting utilities have
documented minimum versions.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Serialized scikit-learn/XGBoost pipelines must be loaded with the recorded
training versions. Runners reject a version mismatch unless the explicit
research-only override is supplied. Because joblib model files use pickle
serialization, load only artifacts that you produced or whose integrity and
provenance you trust.

## Reproducing the controlled benchmark

### 1. Prepare data

```bash
python src/prepare_data.py
```

Place the downloaded upstream files in the following local layout first:

```text
data/raw/adult/adult.data
data/raw/adult/adult.test
data/raw/bank_marketing/bank-full.csv
data/raw/covertype/covtype.data           # .gz or source ZIP also accepted
data/raw/acs_income/psam_p06.csv          # 2018 California one-year PUMS
data/raw/diabetes/diabetic_data.csv
```

The command creates `data/processed/<dataset>/` with frozen training,
calibration, and test-pool splits plus a manifest. Raw datasets are not
downloaded or redistributed by this repository.

### 2. Train calibrated base models

```bash
python experiments/run_phase2.py \
  --data-dir data/processed \
  --results-dir results/base_models
```

The runner is resumable. On a time-limited platform, pass `--max-hours` and
rerun the same command to continue from validated checkpoints.

### 3. Build the ten-seed stream cache

```bash
python experiments/build_stream_cache.py \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --data-dir data/processed \
  --base-models-dir results/base_models \
  --cache-dir results/binary_cache_10_seeds
```

Run the associated cache audit before downstream evaluation. Cached features,
predictions, and oracle arrays allow new monitors to be evaluated on exactly
the same predictor streams without retraining the base models.

### 4. Run and audit Schema 13

The full binary, multiclass, SHD-sensitivity, and final-analysis sequence is
long-running and checkpointed for time-limited compute platforms. Use
`--help` on the relevant runner to inspect its required cache, model, output,
seed, and time-limit arguments. Each completed stage must pass its matching
`audit_*.py` script before the next stage is accepted.

### 5. Recompute final statistics

```bash
python experiments/analyze_step12_schema13.py \
  --binary-dir results/binary_step12_fidelity_results \
  --multiclass-dir results/multiclass_step12_schema13_results \
  --shd-sensitivity-dir results/shd_sensitivity \
  --output-dir results/schema13_final_analysis \
  --bootstrap-replicates 5000 \
  --random-seed 20260927

python experiments/audit_step12_schema13_analysis.py \
  --analysis-dir results/schema13_final_analysis
```

The analysis unit, bootstrap design, multiple-testing correction, and
mixed-effects specifications are encoded in
`experiments/analyze_step12_schema13.py`; its output manifest records the
frozen random seed, replicate count, endpoint separation, and input hashes.

## Adding a monitor

A new monitor should:

1. consume only declared source information, the frozen predictor, and
   unlabeled target observations;
2. declare its estimand and event target before evaluation;
3. use the cached predictor-stream identifier for paired comparisons;
4. keep target-derived oracle quantities outside the monitor module; and
5. add unit tests and a fail-closed output audit.

Unsupported capabilities should be recorded as not applicable, not counted as
failures and not silently imputed.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the required tests, documentation,
and scientific-integrity checks for changes.

## Dataset sources

- [Adult](https://archive.ics.uci.edu/dataset/2/adult)
- [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [ACS Income](https://www.census.gov/programs-surveys/acs/microdata.html)
- [Covertype](https://archive.ics.uci.edu/dataset/31/covertype)
- [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Users remain responsible for complying with each upstream dataset's terms.

## Limitations

- The cross-dataset controlled study is tabular and primarily binary.
- The multiclass extension contains one dataset.
- Controlled interventions do not replace natural temporal or geographic
  validation; the repository includes a separate natural-shift protocol.
- SHD applicability depends on selector feasibility and its published
  assumptions.
- XPE uses a fidelity-audited scalable Shapley approximation in the full run;
  the reference KernelSHAP backend remains available for direct checks.
- Ten seeds improve replication but do not create additional independent
  datasets.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Please cite
the accompanying manuscript when using the benchmark in research. The
metadata will be updated with the final venue and DOI after publication.

## Contact and issue reporting

Use the repository issue tracker for reproducibility questions and bug
reports. Please include the schema version, command, manifest, audit output,
and environment versions; do not attach restricted raw data.

## License

TabMon-Bench is released under the [MIT License](LICENSE). Upstream datasets
retain their original licenses and terms of use.
