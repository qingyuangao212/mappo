import os.path

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
import gym
from gfootball.env import create_environment
import torch
import numpy as np
import os
import time
import argparse

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


model_config = dict(gain=0.01,
                    hidden_size=64,
                    use_orthogonal=True,
                    use_policy_active_masks=True,
                    use_naive_recurrent_policy=False,
                    use_recurrent_policy=True,
                    recurrent_N=1,
                    # for MLPBase
                    use_feature_normalization=True,
                    stacked_frames=1,
                    use_ReLU=True,
                    layer_N=1,
                    )

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--run", type=int, default=10)
args = parser.parse_args()

model_name = args.model_name
model_dir = f'wandb/{model_name}_actor.pt'

# %%
env = create_environment("academy_counterattack_hard",
                         number_of_left_players_agent_controls=4,
                         number_of_right_players_agent_controls=0,
                         representation='simple115v2',
                         rewards='scoring,checkpoints',
                         write_goal_dumps=True,
                         write_full_episode_dumps=True,
                         write_video=True,
                         logdir=os.path.join('results/render', model_name),     # mkdir if not exists
                         render=True
                         )

# copied from FootballEnv
action_space = list(env.action_space)
observation_space = [gym.spaces.Box(low, high) for low, high in
                     zip(env.observation_space.low, env.observation_space.high)]
observation_space = observation_space[0]
action_space = action_space[0]

# print(f"obs space: {observation_space}")
# print(f"action space: {action_space}")

model_config = Struct(**model_config)
actor = R_Actor(model_config, observation_space, action_space, device=device)
policy_actor_state_dict = torch.load(model_dir)
actor.load_state_dict(policy_actor_state_dict)
actor.eval()  # set eval mode


# print(rnn_states.shape)
# runs = 100
# for run in range(runs):
#
#     rnn_states = np.zeros((4, 1, 64), dtype=np.float32)
#     masks = np.ones((4, 1), dtype=np.float32)
#     obs = env.reset()
#     # img = env.unwrapped.observation()[0].keys()
#     dones = False
#     while not dones:
#         with torch.no_grad():
#             action, action_prob, rnn_states = actor(obs, rnn_states, masks, deterministic=True)
#
#         action = np.array(action.to('cpu')).squeeze()
#         obs, rewards, dones, infos = env.step(action)
#         # print(dones)
#         # print(infos)
#         # env.render()
#         # img = env.unwrapped.observation()[0].keys()
#         # print(img)

def run():
    global seed
    print(seed)
    rnn_states = np.zeros((4, 1, 64), dtype=np.float32)
    masks = np.ones((4, 1), dtype=np.float32)
    env.seed(seed)
    obs = env.reset()
    # img = env.unwrapped.observation()[0].keys()
    dones = False
    while not dones:
        with torch.no_grad():
            action, action_prob, rnn_states = actor(obs, rnn_states, masks, deterministic=True)

        action = np.array(action.to('cpu')).squeeze()
        obs, rewards, dones, infos = env.step(action)
    seed = seed + 1
    print(infos)


if __name__ == "__main__":
    import timeit
    import datetime
    num_runs = args.run
    seed = 0
    total_seconds = timeit.timeit(stmt="run()", globals=globals(), number=num_runs)
    print("="*50)
    print(f"Total time: {datetime.timedelta(seconds=total_seconds)}")
    print(f"Average time: {datetime.timedelta(seconds=total_seconds/num_runs)}")
