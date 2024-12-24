#!/usr/bin/env python3

import optuna

def copy_top_trials(
    source_study_name="pretrain_1",
    target_study_name="pretrain_best",
    storage_name="sqlite:///Trials/optuna.db",
    n_top=50
):
    # 1. Load the source study
    source_study = optuna.load_study(
        study_name=source_study_name,
        storage=storage_name
    )
    try:
        optuna.delete_study(
            study_name=target_study_name,
            storage=storage_name
        )
        print(f"Deleted existing study '{target_study_name}'.")
    except KeyError:
        # A KeyError is raised if the study name doesn't exist in the storage
        print(f"No existing study '{target_study_name}' to delete.")
    
    # 2. Create a fresh target study with the same objective direction
    target_study = optuna.create_study(
        study_name=target_study_name,
        direction=source_study.direction,  # Match the original direction ('maximize')
        storage=storage_name,
        load_if_exists=False  # ensures we're creating a brand new one
    )
    
    # 3. Filter out only completed trials that have a recorded objective value
    completed_trials = [
        t for t in source_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    
    # 4. Sort them by objective value (descending) since we're maximizing
    sorted_completed_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    top_trials = sorted_completed_trials[:n_top]
    
    # 5. Copy those top trials into the new study
    for orig_trial in top_trials:
        # Construct a FrozenTrial; keep the original trial number
        new_trial = optuna.trial.FrozenTrial(
            number=orig_trial.number,  # Preserve the original trial number
            state=optuna.trial.TrialState.COMPLETE,
            value=orig_trial.value,
            datetime_start=orig_trial.datetime_start,
            datetime_complete=orig_trial.datetime_complete,
            params=orig_trial.params,
            distributions=orig_trial.distributions,
            user_attrs=orig_trial.user_attrs,
            system_attrs=orig_trial.system_attrs,
            intermediate_values=orig_trial.intermediate_values,
            trial_id=None  # Let the storage assign a new trial ID
        )
        
        # Add the copied trial to the target study
        target_study.add_trial(new_trial)

def main():
    copy_top_trials()

if __name__ == "__main__":
    main()
