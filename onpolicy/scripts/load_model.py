from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
import gym
from gfootball.env import create_environment
import torch
import numpy as np
import time

# from onpolicy.envs.football.environment import FootballEnv

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


model_config = dict(gain=0.010,
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

# %%
env = create_environment("academy_counterattack_hard",
                         number_of_left_players_agent_controls=4,
                         number_of_right_players_agent_controls=0,
                         representation='simple115v2',
                         rewards='scoring,checkpoints',
                         write_goal_dumps=True,
                         write_full_episode_dumps=True,
                         render=True,
                         write_video=True,
                         logdir='./wandb'
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
model_dir = 'wandb/fineTuneCA_15_6_3_actor.pt'
policy_actor_state_dict = torch.load(model_dir)
actor.load_state_dict(policy_actor_state_dict)
actor.eval()  # set eval mode
# obs = env.reset()
# for ob in obs:
#     print(actor(ob))


# print(rnn_states.shape)
runs = 10
for run in range(runs):

    rnn_states = np.zeros((4, 1, 64), dtype=np.float32)
    masks = np.ones((4, 1), dtype=np.float32)
    obs = env.reset()
    dones = False
    while not np.any(dones):
        action, action_prob, rnn_states = actor(obs, rnn_states, masks, deterministic=True)
        action = np.array(action.to('cpu')).squeeze()
        obs, dones, rewards, infos = env.step(action)
        print(dones)
        # env.render()
        # img = env.unwrapped.observation()[0]["frame"]
