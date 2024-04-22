#!/bin/bash
# This bash script trains machine learning models
# It uses various configurations and folds to optimize the models

# declare arrays
declare -a FOLDS=(0 1 2 3)
declare -a CFGNAMES=("cfg_1" "cfg_2a" "cfg_2b" "cfg_3" "cfg_4" "cfg_5a" "cfg_5b" "cfg_5c" "cfg_5d")

# error checking mechanism for the python script's execution
function handle_error() {
    echo "Error occurred while executing the script. Check the parameters or script."
    exit 1
}

# loop for each training fold and configuration
for fold in ${FOLDS[@]}; do
    for run in {1..2}; do  # Running each fold twice
        for cfgname in ${CFGNAMES[@]}; do
            # run the training script
            python train.py -C $cfgname --fold $fold || handle_error
        done
    done
done

# further training stages if desired
for extra_fold in -1 ; do
    for run in {1..8}; do  # Running fullfit 8 times
        for cfgname in ${CFGNAMES[@]}; do
            python train.py -C $cfgname --fold $extra_fold || handle_error
        done
    done
done

echo "Training Completed Successfully!"