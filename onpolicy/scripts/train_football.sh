#!/bin/sh
env="academy_3_vs_1_with_keeper"
rep="simple115v2"
num_left_agents=3
num_right_agents=0

algo="rmappo"
exp="check" # if not set, this is current time
seed_max=1

echo "env is ${env}, representation is ${rep}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --use_valuenorm --env_name ${env} \
    --algorithm_name ${algo} --experiment_name ${exp} --representation ${rep} \
    --number_of_left_players_agent_controls ${num_left_agents} \
    --number_of_right_players_agent_controls ${num_right_agents} --seed ${seed_max} \
    --n_rollout_threads 2 --num_mini_batch 2 --episode_length 200 --num_env_steps 25000000 \
    --ppo_epoch 15 --use_ReLU --wandb_name "football" --user_name "peter_gao" \
    --use_wandb --save_interval 200000 --log_interval 200000 \
    --use_eval --eval_interval 400000 --eval_episodes 100 --n_eval_rollout_threads 100 --cuda false
done