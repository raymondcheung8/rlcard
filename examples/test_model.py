''' Compare different models that go against a random agent
'''
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, plot_bar

import rlcard
import torch

seed_num = 20
num_of_agents = 11  # CHANGE THIS EVERY TIME A NEW TEST AGENT IS ADDED
num_of_games = 10000

# Seed numpy, torch, random
set_seed(seed_num)

# Make environment
envs = [rlcard.make('no-limit-holdem', config={'seed': seed_num}) for _ in range(num_of_agents)]

random_agent = RandomAgent(num_actions=envs[0].num_actions)
test_agents = [torch.load('../experiments/nolimitholdem_dqn_result/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result2/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result3/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result5/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result4/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result8/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result6/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result7/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result9/model.pth'),
               torch.load('../experiments/nolimitholdem_dqn_result10/model.pth'),
               RandomAgent(num_actions=envs[num_of_agents - 1].num_actions)]

dqn_agent_names = [
    'dqn-random_50k',
    'dqn-dqn_50k',
    'dqn-random_20k',
    'dqn-dqn_20k',
    'dqn-random_10k',
    'dqn-dqn_10k',
    'dqn-random_5k',
    'dqn-dqn_5k',
    'dqn-random_1k',
    'dqn-dqn_1k',
    'random'
]

dqn_gain = [0 for _ in range(num_of_agents)]

for i in range(num_of_agents):
    envs[i].set_agents([test_agents[i], random_agent])

    for _ in range(num_of_games):
        trajectories, payoffs = envs[i].run(is_training=False)
        dqn_gain[i] += payoffs[0]

print(f"Test_agents gain {dqn_gain} chips.")
plot_bar(dqn_agent_names, dqn_gain, '../experiments/compare.png', num_of_games)
