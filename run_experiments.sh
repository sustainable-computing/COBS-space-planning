#!/bin/bash

# Step 1 - Create historical data
# Please modify simulation length and other hyperparameters in `building_simulation.py`
python3 building_simulation.py

# Step 2 - Run optimization for long-term occupants
SEEDS=(1 2 3 4 5 6 7 8 9 0)
ESTIMATORS=("rf" "nn" "linear")
TEMPOPTS=("False" "True")
for est in ${ESTIMATORS[@]} ; do
    for seed in ${SEEDS[@]} ; do
        for temp in ${TEMPOPTS[@]} ; do
            tmux new-session -d -s "bash-${seed}-${est}-${temp}" "python3 longterm_optimization.py --seed 3${seed} --estimator ${est} --with_zone_temp ${temp} --warm_start False"
        done
    done
done

# Step 3 - Run baseline assignment strategies for long-term occupants
SEEDS=(1 2 3 4 5 6 7 8 9 0)
BASELINES=("uniform_number" "uniform_ratio" "random")
TEMPOPTS=("False" "True")
for alg in ${BASELINES[@]} ; do
    for seed in ${SEEDS[@]} ; do
        for temp in ${TEMPOPTS[@]} ; do
            tmux new-session -d -s "bash-${seed}-${alg}-${temp}" "python3 longterm_optimization.py --seed 3${seed} --special ${alg} --with_zone_temp ${temp} --warm_start False"
        done
    done
done

# Step 4 - Run online-assignment algorithms for short-term occupants
SEEDS=(1 2 3 4 5 6 7 8 9 0)
BATCHS=(0 1 2 3 4)
ONLINES=("bestfit_energy" "bestfit_space" "online_minlp" "uniform_number" "uniform_ratio" "random")
VISITORS=(100 200)
for est in ${ONLINES[@]} ; do
    for seed in ${SEEDS[@]} ; do
        for vis in ${VISITORS[@]} ; do
            for bat in ${BATCHS[@]} ; do
                tmux new-session -d -s "bash-${seed}-${est}-${vis}-batch-${bat}" "python3 full_experiment.py --curves False --parallel 5 --n ${bat} --reprocess True --designate_base _nn_adjacentzone_True --online ${est} --num_visitor ${vis} --seed ${seed} --job_name s-${seed}-${vis}-${est}-bat${bat}"
            done
        done
    done
done