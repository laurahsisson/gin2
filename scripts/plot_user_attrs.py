#!/usr/bin/env python3

import optuna
import matplotlib.pyplot as plt
import numpy as np

def plot_user_attrs_vs_metric(
    study_name,
    storage_name,
    user_attr_key,
    log_x
):
    # Load the study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )
    
    # Filter trials with completed states and valid user attributes
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    # Extract user_attr_key values and corresponding metrics
    x_values = []
    y_values = []

    for trial in completed_trials:
        if user_attr_key in trial.user_attrs:  # Ensure the user_attr_key exists
            x_values.append(trial.user_attrs[user_attr_key])
            y_values.append(trial.value)
    
    if not x_values:
        print(f"No data found for user_attr_key='{user_attr_key}'.")
        return

    # Plot user_attrs vs metric
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.7)
    plt.xlabel(f"User Attribute: {user_attr_key}")
    if log_x:
        plt.xscale('log')
    plt.ylabel("Objective Value")
    plt.title(f"{user_attr_key} vs Metric ({study_name})")
    plt.grid(True)
    plt.show()

def main():
    plot_user_attrs_vs_metric(
        study_name="pretrain_best",
        storage_name="sqlite:///Trials/optuna.db",
        user_attr_key="model_size",  # Replace with your desired key
        log_x=True
    )

if __name__ == "__main__":
    main()
