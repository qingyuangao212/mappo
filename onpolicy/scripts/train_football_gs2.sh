#!/bin/sh

# grid search 2,  see run_name for params
envs=(
      "academy_counterattack_hard"
      "academy_pass_and_shoot_with_keeper"
      "academy_run_pass_and_shoot_with_keeper"
)

# we control only left agents
list_num_left_agents=(4 2 2)
rep="simple115v2"
num_right_agents=0
algo="rmappo"
seed_max=1
exp="gridSearch2"
run_name="[lr,clip_param]"

device=0
for i in "${!envs[@]}"; do

  env=${envs[i]}
  num_left_agents=${list_num_left_agents[i]}


  echo "env is ${env}, representation is ${rep}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

  # grid search
#  for ppo_epoch in 5 10 15 20; do
#  for num_mini_batch in 1 2 4; do
  for lr in 0.0001 0.0005 0.0008 0.001; do
#  for use_relu in true false; do
  for clip_param in 0.05 0.1 0.15 0.2 0.3 0.5; do
#  for gain in 0.01 1; do
#  for entropy_coef in 0.01 0.015 0.02; do

      for seed in $(seq ${seed_max}); do
      # for ((seed=seed_max; seed>0; seed--))

          # use experiment_name and run_name to describe experiment and run
          # set CUDA_VISIBLE_DEVICES to be remainder of seed devided by number of gpus

          CUDA_VISIBLE_DEVICES=$((device%2)) python3 train/train_football.py --use_valuenorm --env_name ${env} \
          --algorithm_name ${algo} --experiment_name ${exp} --run_name "${run_name}" --representation ${rep} \
          --number_of_left_players_agent_controls ${num_left_agents} \
          --number_of_right_players_agent_controls ${num_right_agents} --seed "${seed}" \
          --n_rollout_threads 50 --num_mini_batch $num_mini_batch --episode_length 400 --num_env_steps 25000000 \
          --ppo_epoch $ppo_epoch --wandb_name "football" --user_name "qingyuan_gao" \
          --use_wandb false --save_interval 100 --log_interval 10 \
          --use_eval --eval_interval 20 --eval_episodes 100 --n_eval_rollout_threads 50 --rewards scoring,checkpoints \
          --lr $lr --clip_param $clip_param &
#          --gain $gain --entropy_coef $entropy_coef &

          ((device += 1))
      done
#  done
#  done
#  done
#  done
#  done
  done
  done
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