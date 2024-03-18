"""
This agent was created using the following videos and adapting them to the rlcard library:

"Proximal Policy Optimization (PPO) is Easy With PyTorch | Full PPO Tutorial"
Tutorial author: Phil Tabor <https://github.com/philtabor>
Tutorial URL: <https://www.youtube.com/watch?v=hlv79rcHws0>
Sourcecode URL: <https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch>

"Let's Code Proximal Policy Optimization"
Tutorial author: Edan Meyer <https://www.youtube.com/@EdanMeyer>
Tutorial URL: <https://www.youtube.com/watch?v=HR8kQMTO8bk>
Sourcecode URL: <https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing>
"""

import os
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='experiments/nolimitholdem_ppo_result/'):
        super(ActorNetwork, self).__init__()

        # Ensure the directory exists before saving the checkpoint
        os.makedirs(os.path.dirname(chkpt_dir), exist_ok=True)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='experiments/nolimitholdem_ppo_result/'):
        super(CriticNetwork, self).__init__()

        # Ensure the directory exists before saving the checkpoint
        os.makedirs(os.path.dirname(chkpt_dir), exist_ok=True)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, target_kl_div=0.01,
                 save_path='experiments/nolimitholdem_ppo_result/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        print(save_path)
        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=save_path)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=save_path)
        self.memory = PPOMemory(batch_size)
        self.target_kl_div = target_kl_div
        self.save_path = save_path
        self.use_raw = False

        self.probsValsList = []

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        # action2, probs, vals = self.choose_action(state['obs'])
        action2, probs, vals = self.probsValsList.pop(0)
        if action != action2:
            print(f"{action} != {action2}")
        self.memory.store_memory(state['obs'], action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def choose_random_action(self, observation, legal_actions):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = random.choice(legal_actions)
        action = T.tensor(action, dtype=T.int).to(self.actor.device)

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def step(self, state):
        legal_actions = list(state['legal_actions'].keys())
        action, probs, value = self.choose_action(state['obs'])

        illegal_action_count = 0
        illegal_action_threshold = 10
        while action not in legal_actions:
            print(f"ACTION: {action}")
            print(f"COUNT: {illegal_action_count}")
            print(f"LEGAL ACTIONS: {legal_actions}")
            action, probs, value = self.choose_action(state['obs'])
            illegal_action_count += 1

            # if the model chooses an illegal action more times than the threshold, choose a random legal action
            if illegal_action_count >= illegal_action_threshold:
                action, probs, value = self.choose_random_action(state['obs'], legal_actions)

        self.probsValsList.append((action, probs, value))
        return action

    def eval_step(self, state):
        action, probs, value = self.choose_action(state['obs'])

        info = {}

        return action, info

    def train(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate the generalized advantage estimations
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # calculate the total_loss from both the policy and value
                total_loss = actor_loss + critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # if model has diverged too much, stop iterating
                kl_div = (old_probs - new_probs).mean()
                if kl_div >= self.target_kl_div:
                    break

        self.memory.clear_memory()
