#!/bin/sh
env="academy_3_vs_1_with_keeper"
rep="simple115v2"
num_left_agents=3
num_right_agents=0
algo="rmappo"
seed_max=1
exp="baseline"

echo "env is ${env}, representation is ${rep}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    # use experiment_name and run_name to describe experiment and run
    # in baseline there's no run, no need to add run_name
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --use_popart --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed "${seed}" \
    --n_rollout_threads 50 --num_mini_batch 2 --episode_length 200 --num_env_steps 25000000 \
    --ppo_epoch 15 --use_ReLU --wandb_name "football" --user_name "qingyuan_gao" \
    --use_wandb --save_interval 100 --log_interval 20 \
    --use_eval --eval_interval 40 --eval_episodes 100 --n_eval_rollout_threads 100 --rewards scoring,checkpoints
done

# num_episodes = num_env_steps // episode_length // n_rollout_threads
# num_train_logs = ( num_episodes // log_iterval ) + 1

# wandb info:
# project: env_name
