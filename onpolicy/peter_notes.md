# Environment

1. Need True for scenario config: `end_episode_on_score`
2. 

# Algorithm
## onpolicy.algorithms.r_mappo.r_mappo
### where do we use algo in runner?
In FootballRunner(football_runner.py)
 - `self.trainer = r_mappo(self.all_args, self.policy, device = self.device)`
 - (base_runner)
    ```
   self.policy = Policy(self.all_args,
                              self.envs.observation_space[0],
                              share_observation_space,    # shared for V and above row for pi?
                              self.envs.action_space[0],
                              device = self.device)```
 - collect():
   - `self.trainer.prep_rollout()`
   ```
   value, action, action_log_prob, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
                                              np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
   ```
trainer is r_mappo.py class, policy is rMAPPOPolicy.py

###Policy
It has actor, critic, actor_optimizer, critic_optimizer

Actor outputs:  
```
actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, 
                                                         available_actions, deterministic)
```
Critic outputs:
values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
#### r_actor_critic.py
Actor: 
- base (CNN or MLP)
- rnn
- act (to compute action and action_logs outputs)