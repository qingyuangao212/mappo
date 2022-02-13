"""
Removed all actions_env from mpe environments
"""
import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


class FootballRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the FootballEnvs. See parent class for
    details. """

    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        assert self.envs.action_space[0].__class__.__name__ == 'Discrete'

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                print('======action shape============')
                print(type(actions))
                print(actions.shape)

                actions_env = actions.squeeze()
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)



                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                print('===========data shape==============')
                [print(x.shape) for x in [obs, rewards, dones, values, actions, action_log_probs, rnn_states, rnn_states_critic]]
                ## error note: dones is (2,) rnn states is (2,3,1,64), need dones to be (2,3)?

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Environment Representation {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.env_name + '_' + self.all_args.representation,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # if self.env_name == "academy_3_vs_1_with_keeper":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):

        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # print('\n', step)
        # print("shapes:========")
        # print(value.shape)
        # print(action.shape)
        # print(action_log_prob.shape)
        # print(rnn_states.shape)
        # print(rnn_states_critic.shape)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic  # substitute actions_env with actions

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))


            # Observe reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    # @torch.no_grad()
    # def render(self):
    #     """Visualize the env."""
    #     envs = self.envs
    #
    #     all_frames = []
    #     for episode in range(self.all_args.render_episodes):
    #         obs = envs.reset()
    #         if self.all_args.save_gifs:
    #             image = envs.render('rgb_array')[0][0]
    #             all_frames.append(image)
    #         else:
    #             envs.render('human')
    #
    #         rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
    #                               dtype=np.float32)
    #         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #
    #         episode_rewards = []
    #
    #         for step in range(self.episode_length):
    #             calc_start = time.time()
    #
    #             self.trainer.prep_rollout()
    #             action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
    #                                                          np.concatenate(rnn_states),
    #                                                          np.concatenate(masks),
    #                                                          deterministic=True)
    #             actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
    #             rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
    #
    #             if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
    #                 for i in range(envs.action_space[0].shape):
    #                     uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
    #                     if i == 0:
    #                         actions_env = uc_actions_env
    #                     else:
    #                         actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
    #             elif envs.action_space[0].__class__.__name__ == 'Discrete':
    #                 actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
    #             else:
    #                 raise NotImplementedError
    #
    #             # Obser reward and next obs
    #             obs, rewards, dones, infos = envs.step(actions_env)
    #             episode_rewards.append(rewards)
    #
    #             rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
    #                                                  dtype=np.float32)
    #             masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #             masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
    #
    #             if self.all_args.save_gifs:
    #                 image = envs.render('rgb_array')[0][0]
    #                 all_frames.append(image)
    #                 calc_end = time.time()
    #                 elapsed = calc_end - calc_start
    #                 if elapsed < self.all_args.ifi:
    #                     time.sleep(self.all_args.ifi - elapsed)
    #             else:
    #                 envs.render('human')
    #
    #         print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
    #
    #     if self.all_args.save_gifs:
    #         imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
