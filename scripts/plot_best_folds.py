import optuna
import matplotlib.pyplot as plt
import numpy as np

def plot_macro_f1_per_epoch_for_top_trials(study_name, storage_name, top_n=5):
    """
    Find the top N trials from the Optuna study and plot macro_f1 per epoch for each fold,
    as well as the average macro_f1 across all folds, in a vertical subplot (5 rows).

    Args:
        study_name (str): Name of the Optuna study.
        storage_name (str): Storage path for the Optuna study.
        top_n (int): Number of top trials to plot.
    """
    # Load the study
    study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )

    # Get the top N trials
    top_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
        reverse=True
    )[:top_n]

    if not top_trials:
        print("No completed trials found in the study.")
        return

    # Create a vertical subplot (5 rows)
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 2 * top_n), sharex=True)

    for i, trial in enumerate(top_trials):
        # Extract fold metrics from the trial's user attributes
        metrics = {
            key: value
            for key, value in trial.user_attrs.items()
            if key.startswith("fold_") and isinstance(value, list)
        }

        if not metrics:
            print(f"No fold metrics found in trial {trial.number}.")
            continue

        # Extract macro_f1 values for each fold
        fold_macro_f1 = {}
        for fold, fold_data in metrics.items():
            fold_macro_f1[fold] = [epoch_data["macro_f1"] for epoch_data in fold_data]

        # Find the number of epochs
        num_epochs = len(next(iter(fold_macro_f1.values())))

        # Calculate average macro_f1 across folds for each epoch
        average_macro_f1 = [
            np.mean([fold_macro_f1[fold][epoch] for fold in fold_macro_f1])
            for epoch in range(num_epochs)
        ]

        # Plot macro_f1 per epoch for each fold
        ax = axes[i] if top_n > 1 else axes
        for fold, macro_f1_values in fold_macro_f1.items():
            ax.plot(range(1, num_epochs + 1), macro_f1_values, label=f"{fold}")

        # Plot the average macro_f1
        ax.plot(
            range(1, num_epochs + 1), average_macro_f1, label="Average", color="black", linewidth=2, linestyle="--"
        )

        # Add labels, title, and legend
        ax.set_ylabel("Macro F1")
        ax.set_title(f"Trial {trial.number}: Macro F1 per Epoch")
        ax.legend()
        ax.grid(True)

    # Add shared x-axis label
    plt.xlabel("Epoch")

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    plot_macro_f1_per_epoch_for_top_trials(
        study_name="pretrain_best",
        storage_name="sqlite:///Trials/optuna.db",
        top_n=4
    )

if __name__ == "__main__":
    main()
