''' Compare different models of ppo_clip that are trained against themselves
'''
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, plot_bar

import rlcard
import torch

seed_num = 1
num_of_games = 10000

test_agents = [torch.load('experiments/nolimitholdem_ppo_result7/model.pth'),
               torch.load('experiments/nolimitholdem_ppo_result8/model.pth'),
               torch.load('experiments/nolimitholdem_ppo_result17/model.pth'),
               torch.load('experiments/nolimitholdem_ppo_result18/model.pth')]

# +1 is for the random agent that gets added later
num_of_agents = len(test_agents) + 1

# Seed numpy, torch, random
set_seed(seed_num)

# Make environment
envs = [rlcard.make('no-limit-holdem', config={'seed': seed_num}) for _ in range(num_of_agents)]

random_agent = RandomAgent(num_actions=envs[0].num_actions)

# Add a random agent to the agents being tested
test_agents.append(RandomAgent(num_actions=envs[num_of_agents - 1].num_actions))

dqn_agent_names = [
    'ppo_c-ppo_c\n1k_a+c',
    'ppo_c-ppo_c\n1k_a+0.5c',
    'ppo_c-ppo_c\n1k_a+0.001c',
    'ppo_c-ppo_c\n1k_a',
    'random'
]

dqn_gain = [0 for _ in range(num_of_agents)]

for i in range(num_of_agents):
    envs[i].set_agents([test_agents[i], random_agent])

    for _ in range(num_of_games):
        trajectories, payoffs = envs[i].run(is_training=False)
        dqn_gain[i] += payoffs[0]

training_method_color = {
    'ppo_c-ppo_c': 'royalblue',
    'random': 'firebrick'
}

print(f"Test_agents gain {dqn_gain} chips.")
plot_bar(dqn_agent_names, dqn_gain, 'experiments/compare_ppo_c-ppo_c.png', num_of_games, training_method_color)
