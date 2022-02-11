#!/bin/sh
env_name="academy_3_vs_1_with_keeper"
representation="simple115v2"
number_of_left_players_agent_controls=3
number_of_left_players_agent_controls=0

algo="rmappo"
exp="check" # if not set, this is current time
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_football.py --use_valuenorm --use_popart --env_name ${env_name} \
    --algorithm_name ${algo} --experiment_name ${exp} --representation ${representation} \
    --number_of_left_players_agent_controls ${number_of_left_players_agent_controls} \
    --number_of_right_players_agent_controls ${number_of_right_players_agent_controls} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 7 --num_mini_batch 1 --episode_length 400 --num_env_steps 20000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "peter_gao" --user_name "peter_gao" \
    --use_wandb
done