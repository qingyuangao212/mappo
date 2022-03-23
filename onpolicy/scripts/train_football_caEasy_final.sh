#!/bin/sh


rep="simple115v2"
num_right_agents=0
algo="rmappo"
seed_max=6
exp="final"


run=0
env="academy_counterattack_easy"
num_left_agents=4
seed_start=4
for seed in 4 5 6; do

    ((run += 1))
    run_name="seed${seed}"
    CUDA_VISIBLE_DEVICES=$((run%2)) python3 train/train_football.py --use_valuenorm --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --run_name "${run_name}" --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed $seed \
    --n_rollout_threads 50 --num_mini_batch 4 --episode_length 200 --num_env_steps 20000000 \
    --ppo_epoch 15 --wandb_name "football" --user_name "qingyuan_gao" \
    --save_interval 100 --log_interval 20 \
    --use_wandb --use_eval --eval_interval 40 --eval_episodes 100 --n_eval_rollout_threads 50 --rewards scoring,checkpoints &
    echo "=================================================="
    echo "run_number_${run}: ${run_name}"

done




# =============== INTERPRETATION =====================
# train a total of {num_env_steps} steps, each episode has max length {episode_length}, use {n_rollout_threads} threads for training and {n_eval_rollout_threads} for eval
# when training : log every {eval_interval} episodes, call eval every {eval_interval}
# eval: run {eval_episodes} number of episodes


# =============== Calculations =====================
# num_episodes = num_env_steps // episode_length // n_rollout_threads
# num_train_logs = ( num_episodes // log_interval ) + 1


# =============== wandb info =====================
# Group: env_name (e.g. academy_3_vs_1_with_keeper) + representation (simple115v2)
# Run: algo_name ("RMAPPO") + experiment_name (e.g. "baseline" or parameter_name) + run_name (e.g. parameter_value)

# Group: representation (simple115v2) + experiment_name (e.g. baseline or some parameter)
# Run: algo_name ("RMAPPO") + run_name (e.g. parameter_value) + seed

# ---UPDATE---
# Now I use only one project name, devide different scenario (env_name) to differnt groups
# I removed representation and algo_name, consider adding it to group name later
# Project: gfootball_mappo
# Group: env_name (academy_3_vs_1_with_keeper) + experiment_name (e.g. baseline or some parameter)
# Run: experiment_name + run_name (e.g. parameter_value) + seed