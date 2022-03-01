import gym
from gym import spaces
import numpy as np
from gfootball.env import create_environment

class FootballEnv(gym.Env):
    """
    Customized gym environment class.


    """


    def __init__(self, args):
        """
        For discrete action space and Box observation space:
            action_space - list of Discrete(): for gym.MultiDiscrete, convert via list(). List length = num_agents
            observation_space: - list of Box(observation_space_n,) space for each agent. List length = num_agents
            share_observation_space: same dimension as observation_space
        """
        # convert args to dict and take only certain keys:
        # this is because create_env will raise error if we pass in irrevelant key-values
        environment_args_dict = {
            'env_name': args.env_name,
            'number_of_left_players_agent_controls': args.number_of_left_players_agent_controls,
            'number_of_right_players_agent_controls': args.number_of_right_players_agent_controls,
            'representation': args.representation,
            'rewards': args.rewards
        }
        if args.env_name in ["academy_3_vs_1_with_keeper", "academy_corner", "academy_counterattack_easy", \
                             "academy_counterattack_hard", "academy_pass_and_shoot_with_keeper", \
                             "academy_run_pass_and_shoot_with_keeper"]:
            # parse args
            self.env = create_environment(**environment_args_dict)
            self.num_left_agents = args.number_of_left_players_agent_controls
            self.num_right_agents = args.number_of_right_players_agent_controls
            self.num_agents = self.num_left_agents + self.num_right_agents
            self.representation = args.representation
            self.rewards = args.rewards

        # you may add additional env init with 'elif' blocks
        else:
            raise NotImplementedError
        assert self.env.action_space.__class__.__name__ == 'MultiDiscrete'
        # football multiDiscrete different from package multiDiscrete

        self.action_space = list(self.env.action_space)

        self.observation_space = [gym.spaces.Box(low, high)
                                  for low, high in zip(self.env.observation_space.low,
                                                       self.env.observation_space.high)]  # dissemble Box(3, 115) into [Box(115,), Box(115,), Box(115,)]

        self.share_observation_space = self.observation_space.copy()

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        """
        Returns:
            obs[np.ndarray]: (num_agents, observation_space_n)
        """
        return self.env.reset()

    def step(self, actions):
        """

        Args:
            actions[np.ndarray]: (num_agents,) for scalar agent action
        Returns:
            obs[np.ndarray]: (num_agents, observation_space_n)
            rewards[np.ndarray]: (num_agents,)
            dones[np.array(dtype: bool)]: (num_agents,)
            info[dict]
        """
        obs, rewards, done, info = self.env.step(actions)
        dones = np.array([done] * self.num_agents)
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        pass
