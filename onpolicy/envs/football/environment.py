import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from gfootball.env import create_environment
from ..mpe.multi_discrete import MultiDiscrete

# ====== sample football config ========
# ENV_CONFIG = dict(
#     env_name="academy_3_vs_1_with_keeper",
#     number_of_left_players_agent_controls=1,
#     number_of_right_players_agent_controls=0,
#     representation="simple115v2",
# )

class FootballEnv(object):

    def __init__(self, args):

        # convert args to dict and take only certain keys:
        # this is because create_env will raise error if we pass in irrevelant key-values
        environment_args = {
            'env_name' : args.env_name,
            'number_of_left_players_agent_controls': args.number_of_left_players_agent_controls,
            'number_of_right_players_agent_controls': args.number_of_right_players_agent_controls,
            'representation': args.representation,
            'rewards': args.rewards
        }
        if args.env_name == "academy_3_vs_1_with_keeper":
            # parse args
            self.env = create_environment(**environment_args)
            self.num_left_agents = args.number_of_left_players_agent_controls
            self.num_right_agents = args.number_of_right_players_agent_controls
            self.num_agents = self.num_left_agents + self.num_right_agents
            self.representation = args.representation
            self.rewards = args.rewards

        # you may add additional env init with 'elif' blocks
        else:
            raise NotImplementedError
        assert self.env.action_space.__class__.__name__=='MultiDiscrete'
        # football multiDiscrete different from package multiDiscrete

        self.action_space = list(self.env.action_space)

        self.observation_space = [gym.spaces.Box(low, high)
                                  for low, high in zip(self.env.observation_space.low, self.env.observation_space.high)]    # dissemble Box(3, 115) into [Box(115,), Box(115,), Box(115,)]

        self.share_observation_space = self.observation_space.copy()

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True): return self.env.reset()

    def step(self, actions: np.ndarray):
        return self.env.step(actions)   # obs, reward, done, info

    def close(self):
        self.env.close()

if __name__ == '__main__':
    pass
