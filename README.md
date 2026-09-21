# Small-Data ML Optimisation of Photocurable Resin Formulations


## How the code works

The script runs the complete modelling workflow:

1. It reads the formulation compositions, monomer properties, and measured Tg values from three CSV files.
2. It calculates composition-weighted physicochemical properties for each formulation.
3. It constructs model features from component weight fractions, aggregated properties, and curing time.
4. It trains a `CatBoostRegressor`. A `+1` monotonic constraint is applied to curing time so that the predicted Tg cannot decrease as curing time increases.
5. It handles left-censored `out_of_range` measurements iteratively. At each EM-like iteration, the unknown Tg is replaced by the conditional mean of a normal distribution given that `Tg < 243.15 K`, and the model is retrained.
6. It generates random candidate formulations, predicts Tg at four curing times, and ranks the candidates lexicographically: first by low early-stage Tg and then by a small early increase in Tg.
7. It saves the trained model, result tables, metrics, and plots to the output directory.

## Installation

Python 3.10 or newer is required.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

By default, the input files are read from `training_data/`. The files use semicolons as field separators and commas as decimal separators.

### `formulations_wt_percent.csv`

Each row represents one formulation. The `Sample_ID` column is required; the remaining columns contain component names and their weight percentages. For example:

```text
Sample_ID;TMPTMA;HEMA;PEGDA;8811;TPOL
sample_001;20,0;30,0;25,0;22,0;3,0
```

The current candidate generator treats the first four component columns as the monomers to be optimised and expects a column named `TPOL` for the photoinitiator.

### `monomer_properties.csv`

The `Component` column must contain the same component names as the formulation table. All other numeric columns are treated as material properties. The script calculates their composition-weighted averages and uses them as model features.

Every component present in `formulations_wt_percent.csv` must have a corresponding row in this file.

### `targets_Tg_conversion.csv`

Required columns:

- `Sample_ID` — sample identifier from the formulation table;
- `Cure_time_s` — curing time in seconds;
- `Tg_K` — measured Tg in kelvin or the string `out_of_range` for a value below the measurement limit.

Rows containing `out_of_range` are treated as left-censored observations. The censoring threshold can be changed with `--censoring-threshold-k`.

## Running the workflow

Run the model with the article parameters:

```bash
python ml_formulations.py --data-dir training_data --output-dir results
```

Run a quick test with fewer trees, EM iterations, and candidates:

```bash
python ml_formulations.py --catboost-iterations 50 --em-iterations 1 --candidates 1000 --quiet
```

Display all command-line options:

```bash
python ml_formulations.py --help
```

Example with custom prediction times and search size:

```bash
python ml_formulations.py --times 5 10 15 --candidates 300000 --top-n 10
```

The fourth prediction time is automatically set to the maximum `Cure_time_s` in the dataset.

## Default model settings

| Parameter | Default | Purpose |
|---|---:|---|
| `sigma_k` | 8.2 K | Standard deviation used by the censoring model |
| `censoring_threshold_k` | 243.15 K | Upper limit for `out_of_range` observations (-30 °C) |
| `em_iterations` | 7 | Number of pseudo-target reconstruction iterations |
| `loss_function` | RMSE | CatBoost loss function |
| `catboost_iterations` | 2000 | Number of boosting iterations |
| `learning_rate` | 0.03 | CatBoost learning rate |
| `depth` | 8 | Tree depth |
| `l2_leaf_reg` | 3.0 | L2 leaf regularisation |
| `random_seed` | 42 | Reproducible data splitting and candidate generation |
| `monotone_constraints` | `Cure_time_s: +1` | Enforces non-decreasing Tg with curing time |

The main model settings are available as command-line arguments. The effective configuration, feature list, and dataset statistics are also written to `results/summary.json` after every run.

## Treatment of censored measurements

The `out_of_range` label means that the actual Tg is below the instrument's lower measurement limit, not that the value is missing at random.

The script first trains CatBoost using only exact observations. It then performs the following EM-like procedure:

1. Predict the mean Tg for every row.
2. Keep measured values unchanged.
3. Replace each censored value with the conditional expectation of a normally distributed target below the censoring threshold.
4. Retrain the model using the exact and reconstructed targets.
5. Repeat for the configured number of iterations.

The reconstructed values are modelling intermediates and should not be interpreted as new experimental measurements.

## Candidate generation and ranking

The default run generates 200,000 candidate formulations. The TPOL fraction is sampled uniformly between 1.5 and 4.0 wt%, and the four monomers share the remaining mass. The nominal bounds for each monomer are 5–80 wt%.

Candidates are ranked by minimising two objectives:

1. predicted Tg at the first curing time (`Tg_T1`);
2. the early increase `Tg(T3) - Tg(T1)`.

Penalties are added for violating the composition bounds and for being too close to an already measured formulation. The default minimum L1 distance is 18 wt%.

The ranking table also reports `nearest_training_L1`, which makes it possible to inspect how far each proposed composition lies from the nearest training formulation.

## Outputs

The following files are created in `results/`:

- `tg_catboost_model.cbm` — trained CatBoost model;
- `summary.json` — run configuration, feature list, dataset statistics, and validation metric;
- `validation_predictions.csv` — predictions for the fixed exact-observation validation split;
- `feature_importance.csv` — built-in CatBoost feature importance;
- `permutation_importance.csv` — permutation impact on MAE;
- `training_formulation_ranking.csv` — ranking of measured formulations;
- `candidate_top.csv` — requested number of top candidate formulations;
- `candidate_top500.csv` — the top 500 candidates with penalties and diagnostic fields;
- `feature_importance.png` — CatBoost feature-importance plot;
- `permutation_importance.png` — permutation-importance plot;
- `candidate_selection.png` — training formulations, Pareto front, target region, and top candidates.

## Validation metric caveat

`validation_mae_exact_only_K` is provided as a reproducibility check for the current algorithm. The initial model is fitted using the training part of an exact-observation split. However, the subsequent EM-like iterations refit the final model using all rows, including the exact values in the validation subset.

The reported MAE must therefore not be interpreted as an independent estimate of generalisation performance. A separate grouped or external validation procedure should be used for model assessment and publication-level performance claims.

## Reproducibility

Candidate generation and data splitting are reproducible when the input data, library versions, and `--seed 42` remain unchanged. The effective settings are recorded in `summary.json`.

CatBoost is run with `allow_writing_files=False`, so it does not create a `catboost_info` directory next to the script.

The original Colab notebook is not required to run this command-line version. `ml_formulations.py` contains no `google.colab` imports, interactive upload calls, notebook shell commands, or run-time package installation steps.

