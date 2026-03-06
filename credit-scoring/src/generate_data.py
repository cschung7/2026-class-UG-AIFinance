"""
Generate realistic synthetic credit scoring data.
No external download needed - fully self-contained.
"""

import numpy as np
import pandas as pd
import os

def generate_credit_data(n_samples=10000, seed=42):
    """Generate synthetic credit data inspired by 'Give Me Some Credit' dataset."""
    np.random.seed(seed)

    # --- Feature generation ---
    age = np.random.normal(45, 12, n_samples).clip(21, 80).astype(int)
    monthly_income = np.random.lognormal(8.5, 0.8, n_samples).clip(1000, 500000).astype(int)
    debt_ratio = np.random.exponential(0.3, n_samples).clip(0, 5).round(4)
    credit_lines = np.random.poisson(8, n_samples).clip(1, 40)
    num_dependents = np.random.poisson(0.8, n_samples).clip(0, 10)

    # Times 30-59 days late
    late_30_59 = np.random.choice(
        [0, 0, 0, 0, 0, 1, 1, 2, 3, 5],
        size=n_samples,
        p=[0.45, 0.15, 0.1, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01]
    )

    # Times 60-89 days late (correlated with 30-59)
    late_60_89 = np.where(
        late_30_59 > 0,
        np.random.binomial(late_30_59, 0.4),
        np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    )

    # Times 90+ days late (correlated with 60-89)
    late_90_plus = np.where(
        late_60_89 > 0,
        np.random.binomial(late_60_89, 0.5),
        np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])
    )

    # Revolving utilization
    revolving_util = np.random.beta(2, 5, n_samples).clip(0, 1.5).round(4)

    # Real estate loans
    real_estate_loans = np.random.poisson(1.0, n_samples).clip(0, 10)

    # --- Target: default probability ---
    logit = (
        -3.0
        + 0.8 * late_90_plus
        + 0.5 * late_60_89
        + 0.3 * late_30_59
        + 1.5 * (debt_ratio > 1).astype(float)
        + 1.2 * (revolving_util > 0.8).astype(float)
        - 0.02 * (age - 45)
        - 0.3 * np.log1p(monthly_income / 10000)
        + np.random.normal(0, 0.5, n_samples)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = (np.random.uniform(0, 1, n_samples) < prob_default).astype(int)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "default": default,
        "revolving_utilization": revolving_util,
        "age": age,
        "times_30_59_days_late": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_income": monthly_income,
        "num_open_credit_lines": credit_lines,
        "times_90_plus_days_late": late_90_plus,
        "num_real_estate_loans": real_estate_loans,
        "times_60_89_days_late": late_60_89,
        "num_dependents": num_dependents,
    })

    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate_credit_data()
    output_path = os.path.join(data_dir, "credit_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} samples")
    print(f"Default rate: {df['default'].mean():.1%}")
    print(f"Saved to: {output_path}")
