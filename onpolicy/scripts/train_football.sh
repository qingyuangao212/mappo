#!/bin/sh
env_name="academy_3_vs_1_with_keeper"
representation="simple115v2"
number_of_left_players_agent_controls=3
number_of_right_players_agent_controls=0

algo="rmappo"
exp="check" # if not set, this is current time
seed_max=1

echo "env is ${env_name}, scenario is ${env_name + '_' + representation}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --use_valuenorm --use_popart --env_name ${env_name} \
    --algorithm_name ${algo} --experiment_name ${exp} --representation ${representation} \
    --number_of_left_players_agent_controls ${number_of_left_players_agent_controls} \
    --number_of_right_players_agent_controls ${number_of_right_players_agent_controls} --seed ${seed} \
    --n_rollout_threads 50 --num_mini_batch 2 --episode_length 200 --num_env_steps 25000000 \
    --ppo_epoch 15 --use_ReLU --wandb_name "football" --user_name "peter_gao" \
    --use_wandb --save_interval 200000 --log_interval 200000 \
    --use_eval --eval_interval 400000 --eval_episodes 100 --n_eval_rollout_threads 100
done