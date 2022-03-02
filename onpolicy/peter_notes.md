# Environment

1. Need True for scenario config: `end_episode_on_score`
2. 

# Algorithm
## onpolicy.algorithms.r_mappo.r_mappo

### where do we use algo in runner?
In FootballRunner(football_runner.py)
 - `self.trainer = r_mappo(self.all_args, self.policy, device = self.device)`
 - (base_runner)
    ```python
   self.policy = Policy(self.all_args,
                              self.envs.observation_space[0],
                              share_observation_space,    # shared for V and above row for pi?
                              self.envs.action_space[0],
                              device = self.device)```
 - collect():
   - `self.trainer.prep_rollout()`
   ```python
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
`values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
`
#### r_actor_critic.py
R_actor.forward(): 
- base (CNN or MLP)
- rnn
- act (to compute action and action_logs outputs)
```python
actor_features = self.base(obs)
actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
```
# Buffer
### `recurrent_generator()` (perform one train)
   ```python
   batch_size = n_rollout_threads * episode_length * num_agents
   data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
   mini_batch_size = data_chunks // num_mini_batch
   ```
each ppo update uses data (mini_batch_size * data_chunks) steps. 
We need **data_chunk** because we need consecutive steps to train the RNN

trainer.train() (`r_mappo.py`) calls `data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)`