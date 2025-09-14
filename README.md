
# Diabetes Disease Progression — Statistical Analysis (OLS + CV)

Author: Harleen Dhatt

**Author:** You (B.Sc. (Honours) Math & Stats, McMaster)  
**Dataset:** Real-world *Diabetes* dataset (scikit-learn). Outcome is a quantitative measure of disease progression one year after baseline.

## Objectives
1. Clean and explore the data.
2. Fit an interpretable **Ordinary Least Squares** regression with statistical inference (t-tests, 95% CIs).
3. Validate with **5-fold cross-validated R²**.
4. Compare with **Ridge** and **Lasso**.
5. Provide clear visuals and diagnostics.

## Key Results

**Test RMSE (OLS):** 53.85  
**Test R² (OLS):** 0.453  

**Cross-validated R² (5-fold, LinearRegression):** 0.478 ± 0.085

**Ridge (α=1.0)** – RMSE: 55.47, R²: 0.419  
**Lasso (α=0.01)** – RMSE: 53.65, R²: 0.457


### Statistically Significant Predictors (α = 0.05)
| variable   |    Coef. |    [0.025 |    0.975] |
|:-----------|---------:|----------:|----------:|
| const      |  151.346 |   145.638 |  157.053  |
| bmi        |  542.429 |   391.062 |  693.795  |
| bp         |  347.704 |   207.35  |  488.057  |
| s5         |  736.199 |   357.689 | 1114.71   |
| sex        | -241.964 |  -376.836 | -107.093  |
| s1         | -931.489 | -1818.84  |  -44.1337 |

> Interpretation (plain language): The model identifies several baseline features that are **significantly associated** with future disease progression. The goodness-of-fit and cross-validated results indicate the model captures meaningful structure without severe overfitting.

## Methods (Brief)
- **Model:** OLS with intercept, fit on an 80/20 train/test split.
- **Inference:** Coefficient t-tests with 95% confidence intervals from `statsmodels`.
- **Validation:** 5-fold R² via scikit-learn.
- **Comparators:** Ridge and Lasso (standard regularizers) on the same split.
- **Diagnostics:** Residuals-vs-fitted and QQ plot.

## Repository Structure
```
project1_diabetes/
├── diabetes_clean.csv
├── coefficients_ols.csv
├── correlation_matrix.csv
├── fig_correlation_matrix.png
├── fig_residuals_vs_fitted.png
├── fig_qqplot_residuals.png
├── fig_ols_coefficients.png
└── analysis_script.py
```

## How to Run Locally
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
python analysis_script.py
```

## Admissions-Ready Highlights (add to CV / SOP)
- *Led an end-to-end statistical analysis on a real clinical dataset:* data cleaning, OLS inference, cross-validated evaluation, and regularization comparison.
- *Produced interpretable findings with confidence intervals and rigorous diagnostics.*
- *Reproducible code and report hosted on GitHub.*

## Figures
(See PNGs in this folder.)
