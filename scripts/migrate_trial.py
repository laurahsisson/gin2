#!/usr/bin/env python3
from tqdm import tqdm
import optuna

def migrate_trials(
    source_study_name="pretrain_1",
    target_study_name="pretrain_2",
    storage_name="sqlite:///Trials/optuna.db",
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
    
    # 5. Copy all trials into the new study
    for orig_trial in tqdm(source_study.trials):
        # Construct a FrozenTrial; keep the original trial number
        new_trial = new_trial = optuna.trial.FrozenTrial(
            number=orig_trial.number,
            state=orig_trial.state,
            value=orig_trial.value,
            datetime_start=orig_trial.datetime_start,
            datetime_complete=orig_trial.datetime_complete,
            params={**orig_trial.params, "do_global_node": False},
            distributions={**orig_trial.distributions, "do_global_node": optuna.distributions.CategoricalDistribution([True, False])},
            user_attrs=orig_trial.user_attrs | {"runtime":"L4"},
            system_attrs=orig_trial.system_attrs,
            intermediate_values=orig_trial.intermediate_values,
            trial_id=None
        )

        
        # Add the copied trial to the target study
        target_study.add_trial(new_trial)

def main():
    migrate_trials()

if __name__ == "__main__":
    main()
