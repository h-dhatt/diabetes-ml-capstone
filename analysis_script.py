
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

def main():
    os.makedirs("project1_diabetes", exist_ok=True)
    base_dir = "project1_diabetes"

    diab = load_diabetes(as_frame=True)
    df = diab.frame.copy()
    df.rename(columns={"target": "disease_progression"}, inplace=True)
    df.to_csv(os.path.join(base_dir, "diabetes_clean.csv"), index=False)

    X = df.drop(columns=["disease_progression"])
    y = df["disease_progression"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_sm = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_train_sm).fit()

    X_test_sm = sm.add_constant(X_test, has_constant="add")
    y_pred = model.predict(X_test_sm)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    coef_table = model.summary2().tables[1].reset_index().rename(columns={"index":"variable"})
    coef_table.to_csv(os.path.join(base_dir, "coefficients_ols.csv"), index=False)

    lin = LinearRegression()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lin, X, y, scoring="r2", cv=cv)

    ridge = Ridge(alpha=1.0, random_state=42).fit(X_train, y_train)
    lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000).fit(X_train, y_train)

    ridge_pred = ridge.predict(X_test)
    lasso_pred = lasso.predict(X_test)

    # Plots
    corr = df.corr(numeric_only=True)
    plt.figure()
    plt.imshow(corr, interpolation='nearest')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix (Diabetes Dataset)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_correlation_matrix.png"), dpi=200)
    plt.close()

    residuals = y_test - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals, s=18)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Fitted values (test)")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted (OLS, Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_residuals_vs_fitted.png"), dpi=200)
    plt.close()

    sm.qqplot(residuals, line="45", fit=True)
    plt.title("QQ Plot of Residuals (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_qqplot_residuals.png"), dpi=200)
    plt.close()

    coef_no_const = coef_table[coef_table["variable"] != "const"]
    plt.figure()
    plt.bar(coef_no_const["variable"], coef_no_const["Coef."].astype(float))
    plt.xticks(rotation=90)
    plt.title("OLS Coefficients")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "fig_ols_coefficients.png"), dpi=200)
    plt.close()

    print("Test RMSE (OLS):", round(rmse, 2))
    print("Test R^2 (OLS):", round(r2, 3))
    print("CV R^2 mean ± std:", round(cv_scores.mean(), 3), "±", round(cv_scores.std(), 3))

if __name__ == "__main__":
    main()
