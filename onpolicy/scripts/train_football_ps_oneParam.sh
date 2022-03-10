#!/bin/sh
envs=(
      "academy_pass_and_shoot_with_keeper"
)

rep="simple115v2"
num_right_agents=0
algo="rmappo"
seed=1
exp="OneParamSearch"


run=0   # used for assigning gpu_device and wait for multiprocessing end

env="academy_pass_and_shoot_with_keeper"
num_left_agents=2

for lr in 0.0001 0.00018182 0.00026364 0.00034545 0.00042727 0.00050909 0.00059091 0.00067273 0.00075455 0.00083636 0.00091818 0.001; do
    ((run += 1))
    run_name="lr_${lr}"

    CUDA_VISIBLE_DEVICES=$((run%2)) python3 train/train_football.py --use_valuenorm --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --run_name "${run_name}" --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed "${seed}" \
    --n_rollout_threads 50 --num_mini_batch 4 --episode_length 200 --num_env_steps 20000000 \
    --ppo_epoch 20 --wandb_name "football" --user_name "qingyuan_gao" \
    --save_interval 100 --log_interval 10 \
    --lr $lr \
    --use_wandb --use_eval --eval_interval 20 --eval_episodes 100 --n_eval_rollout_threads 50 --rewards scoring,checkpoints &
    echo "=================================================="
    echo "run_number_${run}: ${run_name}"
    if (( run % 12 == 0 ));
      then wait
    fi
done

for entropy_coef in 0.005 0.00909091 0.01318182 0.01727273 0.02136364 0.02545455 0.02954545 0.03363636 0.03772727 0.04181818 0.04590909 0.05; do
    ((run += 1))
    run_name="entropyCoef_${entropy_coef}"

    CUDA_VISIBLE_DEVICES=$((run%2)) python3 train/train_football.py --use_valuenorm --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --run_name "${run_name}" --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed "${seed}" \
    --n_rollout_threads 50 --num_mini_batch 4 --episode_length 200 --num_env_steps 20000000 \
    --ppo_epoch 20 --wandb_name "football" --user_name "qingyuan_gao" \
    --save_interval 100 --log_interval 10 \
    --entropy_coef $entropy_coef \
    --use_wandb --use_eval --eval_interval 20 --eval_episodes 100 --n_eval_rollout_threads 50 --rewards scoring,checkpoints &
    echo "=================================================="
    echo "run_number_${run}: ${run_name}"
    if (( run % 12 == 0 ));
      then wait
    fi
done

for gain in 0.005 0.00909091 0.01318182 0.01727273 0.02136364 0.02545455 0.02954545 0.03363636 0.03772727 0.04181818 0.04590909 0.05; do
    ((run += 1))
    run_name="gain_${gain}"

    CUDA_VISIBLE_DEVICES=$((run%2)) python3 train/train_football.py --use_valuenorm --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --run_name "${run_name}" --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed "${seed}" \
    --n_rollout_threads 50 --num_mini_batch 4 --episode_length 200 --num_env_steps 20000000 \
    --ppo_epoch 20 --wandb_name "football" --user_name "qingyuan_gao" \
    --save_interval 100 --log_interval 10 \
    --gain $gain \
    --use_wandb --use_eval --eval_interval 20 --eval_episodes 100 --n_eval_rollout_threads 50 --rewards scoring,checkpoints &
    echo "=================================================="
    echo "run_number_${run}: ${run_name}"
    if (( run % 12 == 0 ));
      then wait
    fi
done

for clip_param in 0.01 0.05 0.1 0.15 0.2 0.25 0.3; do
    ((run += 1))
    run_name="clipParam_${clip_param}"

    CUDA_VISIBLE_DEVICES=$((run%2)) python3 train/train_football.py --use_valuenorm --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --run_name "${run_name}" --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed "${seed}" \
    --n_rollout_threads 50 --num_mini_batch 4 --episode_length 200 --num_env_steps 20000000 \
    --ppo_epoch 20 --wandb_name "football" --user_name "qingyuan_gao" \
    --save_interval 100 --log_interval 10 \
    --clip_param $clip_param \
    --use_wandb --use_eval --eval_interval 20 --eval_episodes 100 --n_eval_rollout_threads 50 --rewards scoring,checkpoints &
    echo "=================================================="
    echo "run_number_${run}: ${run_name}"
    if (( run % 12 == 0 ));
      then wait
    fi
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