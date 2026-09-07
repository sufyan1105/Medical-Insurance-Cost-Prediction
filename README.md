# Medical Insurance Cost Predictor

Predicting annual medical insurance charges from six demographic and health attributes, using scikit-learn regression models.

The headline result is not the model — it is a single engineered feature. A `bmi × smoker` interaction term, derived from one scatter plot, accounts for **84% of the Random Forest's predictive work** and closes most of the gap between a plain linear regression and a gradient-boosted ensemble.

**Tech:** Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

---

## Problem

Given six facts about a person — age, sex, BMI, number of dependents, smoking status and US region — estimate their annual insurance charge in USD. This is a supervised regression task on tabular data.

## Dataset

`insurance.csv` — 1,338 rows, 7 columns, **no missing values**.

| Feature | Description | Type |
|---|---|---|
| `age` | Age of the primary beneficiary (18–64) | Numerical |
| `sex` | male / female | Binary categorical |
| `bmi` | Body Mass Index (15.96–53.13) | Numerical |
| `children` | Number of dependents covered (0–5) | Discrete numerical |
| `smoker` | yes / no | Binary categorical |
| `region` | northeast / northwest / southeast / southwest | Categorical |
| **`charges`** | **Annual insurance cost in USD — target** | **Continuous** |

Target distribution:

| Statistic | Value |
|---|---|
| Mean | $13,270 |
| Median | $9,382 |
| Std | $12,110 |
| Min | $1,122 |
| Max | $63,770 |

Mean far exceeds median — the target is heavily right-skewed. More importantly, it is **bimodal**: two separate populations rather than one skewed one. Class balance matters here too — only 274 of 1,338 people (20.5%) are smokers.

---

## Approach

The notebook runs in four phases.

**Phase 1 — Loading & inspection.** Shape, dtypes, summary statistics, null audit, cardinality of each categorical column. The cardinality check drives the encoding strategy in Phase 3.

**Phase 2 — Exploratory data analysis.** Target distribution (raw and log-transformed), KDE of charges split by smoking status, correlation heatmap, pairplot, BMI-vs-charges and age-vs-charges scatter plots coloured by smoker, box plots for each categorical, and a regional breakdown with a confound check.

**Phase 3 — Preprocessing & feature engineering.** Binary mapping for `sex` and `smoker`; one-hot encoding with `drop_first=True` for `region`; three engineered features:

```python
df_clean["bmi_smoker"]  = df_clean["bmi"] * df_clean["smoker"]
df_clean["age_smoker"]  = df_clean["age"] * df_clean["smoker"]
df_clean["age_squared"] = df_clean["age"] ** 2
```

**Phase 4 — Modelling & evaluation.** Five models, hyperparameter sweeps, four metrics, feature-importance and coefficient inspection, residual analysis, and 5-fold cross-validation.

Scaling is handled by `make_pipeline(StandardScaler(), model)` for the linear family, so the scaler is refitted inside every cross-validation fold rather than once on the full dataset. Tree models take raw features.

---

## Key findings

### 1. The smoker interaction is the model

BMI barely affects charges for non-smokers, but drives them sharply for smokers — visible as two distinct clusters in the BMI-vs-charges scatter, separating near the clinical obesity threshold of BMI 30. A linear model with one global `bmi` slope cannot express that. The `bmi_smoker` interaction gives it a second slope that switches on only for smokers.

Random Forest feature importances:

| Feature | Importance |
|---|---|
| `bmi_smoker` | **0.8398** |
| `age` | 0.0561 |
| `age_squared` | 0.0523 |
| `age_smoker` | 0.0227 |
| `bmi` | 0.0133 |
| `children` | 0.0091 |
| `smoker` | 0.0031 |
| `region_*` (3 cols) | 0.0029 combined |
| `sex` | 0.0007 |

**`smoker` scoring 0.0031 does not mean smoking is irrelevant.** `bmi_smoker` is zero for every non-smoker, so splitting on it already identifies smokers — `smoker` is redundant *given a feature engineered from it*. Importance scores divide credit among correlated features; they do not measure standalone effect. Dropping `bmi_smoker` and refitting sends `smoker` straight to the top.

The same caveat appears twice more in this project: Lasso zeroes `age` while keeping `age_squared` (they correlate at ~0.99), and the linear model gives `smoker` a large *negative* coefficient that is only interpretable when added to `bmi_smoker × bmi`. **Once an interaction is in the model, its parent's coefficient cannot be read alone.**

### 2. Regularisation contributed nothing predictive

Ridge and Lasso land within 0.0001 R² of plain OLS in cross-validation. With 11 features and 1,338 rows there is nothing for a penalty to fix.

Lasso was still useful *diagnostically*. Sweeping alpha and watching which features die reveals a ranking derived from a completely different mechanism than the forest's:

| alpha | Test R² | Features zeroed | Survivors |
|---|---|---|---|
| 10 | 0.8676 | 2 / 11 | all but `age`, `age_smoker` |
| 100 | 0.8626 | 3 / 11 | — |
| 500 | 0.8438 | 7 / 11 | `bmi_smoker`, `age_squared`, `bmi`, `children` |
| 1000 | 0.8278 | 8 / 11 | `bmi_smoker`, `age_squared`, `bmi` |

Three features retain R² 0.83. `sex` and the region dummies fall out first — independently confirming what the box plots suggested.

### 3. Overfitting, measured

The Random Forest depth sweep is a clean demonstration of model capacity:

| max_depth | Train R² | Test R² | Gap |
|---|---|---|---|
| 3 | 0.8492 | 0.8519 | −0.0027 |
| 4 | 0.8678 | 0.8683 | −0.0005 |
| 5 | 0.8782 | 0.8713 | 0.0069 |
| **6** | 0.8895 | **0.8735** | 0.0161 |
| 8 | 0.9232 | 0.8715 | 0.0518 |
| 10 | 0.9523 | 0.8684 | 0.0839 |
| None | 0.9739 | 0.8652 | 0.1087 |

Unrestricted trees reach 0.974 on training data and *lose* 0.008 on test. Every point of that extra training accuracy is memorisation — and it is invisible if you look only at training performance, where `None` appears to be the best model in the table.

### 4. Results

Single 80/20 split (`random_state=42`):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Gradient Boosting** | **$4,287.71** | **$2,417.17** | **0.8816** |
| Random Forest | $4,393.42 | $2,519.39 | 0.8757 |
| Ridge | $4,529.94 | $2,764.58 | 0.8678 |
| Lasso | $4,533.31 | $2,763.70 | 0.8676 |
| Linear Regression | $4,538.39 | $2,757.34 | 0.8673 |

5-fold cross-validation (shuffled, `random_state=42`):

| Model | CV R² | Per-fold |
|---|---|---|
| **Gradient Boosting** | **0.8574 ± 0.0339** | 0.882, 0.856, 0.902, 0.803, 0.845 |
| Random Forest | 0.8538 ± 0.0335 | 0.879, 0.849, 0.899, 0.801, 0.841 |
| Linear Regression | 0.8366 ± 0.0371 | 0.867, 0.841, 0.876, 0.771, 0.829 |
| Lasso | 0.8366 ± 0.0375 | 0.868, 0.841, 0.876, 0.770, 0.830 |
| Ridge | 0.8365 ± 0.0375 | 0.868, 0.841, 0.875, 0.770, 0.829 |

### 5. Reading those numbers honestly

**The single split was optimistic by ~0.024 R².** Gradient Boosting scores 0.8816 on the held-out set but 0.8574 across five folds. That difference is which 268 rows happened to be sealed away. The cross-validated figure is the one to quote.

**Fold-to-fold std (±0.034) is ten times the gap between the top two models (0.0036).** On means and error bars alone, Gradient Boosting and Random Forest are indistinguishable.

**But the folds are paired, which changes the analysis.** The ±0.034 measures how much *fold difficulty* varies, not how uncertain the model difference is — fold 4 is hard for every model (0.771 / 0.801 / 0.803), inflating all five standard deviations identically. Since every model faced the same five folds, the correct comparison is *within* fold, where Gradient Boosting wins 5/5 and the tree family beats the linear family 5/5.

So the defensible conclusion has two parts:

- Tree ensembles genuinely beat the linear family — about +0.02 R², consistent in every fold. Real, and modest.
- Gradient Boosting vs Random Forest: consistent direction, negligible magnitude (0.003–0.007). Effectively tied.

Claiming Gradient Boosting "won" on a 0.006 margin would not survive review.

### 6. Where the model fails, and why it cannot be fixed

Residual analysis of the best model shows two systematic patterns.

**A small upward bias on ordinary cases.** The residual cloud sits at roughly −$1,000 to −$3,000 rather than centred on zero — the model over-quotes the typical customer. This is squared-error loss behaving as designed: RMSE punishes large misses quadratically, so the model hedges upward everywhere to limit damage from the expensive cases it cannot anticipate.

**Large under-predictions on a specific subgroup.** Residuals of +$5,000 to +$21,000, clustered at *low* predicted values. These are non-smokers with unremarkable BMI who were billed $28,000 when the model quoted $8,000. Nothing in the six available columns explains them.

They are visible from Phase 2 onward as a middle band in the age-vs-charges scatter containing both smokers and non-smokers. They are the reason every model — from a straight line to a boosted ensemble — stops near R² 0.86. That ceiling is a property of the dataset, not of the algorithms: six demographic columns cannot predict a cancer diagnosis or a car accident.

Notably, residual spread does **not** widen with prediction magnitude. The classic heteroscedasticity pattern is absent, which is why a log transform of the target was tested and rejected — logging fixes variance that scales with the prediction, not contamination from a second, unpredictable population.

### 7. Practical reading

MAE of $2,417 against a mean charge of $13,270 is roughly **18% error on a typical bill**.

That is adequate for pricing a portfolio, where errors in both directions cancel. It is not adequate for quoting an individual — you would be off by a fifth of the price on a routine case and by far more on an unusual one.

---

## Limitations

- **Hyperparameters were swept against the test set.** The Ridge alpha, Random Forest depth and leaf sweeps all read test scores to pick a value, which is a mild form of peeking — repeated enough times, the test score stops being an honest estimate of unseen performance. The correct approach is `GridSearchCV` with an internal validation split, touching the test set only once at the end. With two knobs the practical impact here is small, but the methodology is worth stating rather than hiding.
- **`age_smoker` did not earn its place.** Of two engineered interactions, one drove 84% of the model and the other contributed a coefficient of 53.4. Feature engineering is hypothesis-driven and some hypotheses fail.
- **1,338 rows is small.** Fold-to-fold variance of ±0.034 R² is a direct consequence, and it limits how finely any two models can be distinguished.
- **No deployment artefact.** The notebook stops at evaluation; there is no serialised model or prediction function for a new individual.

## Possible extensions

- `GridSearchCV` / `RandomizedSearchCV` over a full pipeline, replacing the manual sweeps
- `permutation_importance` on held-out data, avoiding the impurity bias that inflates continuous features like `bmi_smoker`
- SHAP values for per-prediction explanations
- XGBoost / LightGBM as additional candidates
- A `predict_charge(age, sex, bmi, children, smoker, region)` helper plus a pickled pipeline

---

## Running it

```bash
git clone https://github.com/sufyan1105/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction
pip install -r requirements.txt
jupyter notebook medical_insurance_cost_predictor.ipynb
```

Place `insurance.csv` in the repository root. Run cells top to bottom — Phase 3 cells are order-dependent and mutate `df_clean` in place.

**Requirements**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Author

**Sufyan Arshad Kadiwala**
[GitHub](https://github.com/sufyan1105) · [LinkedIn](https://www.linkedin.com/in/sufyan-arshad-kadiwala-a35717290)
