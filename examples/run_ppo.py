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
import numpy as np
import matplotlib.pyplot as plt
import rlcard
import argparse
from rlcard.agents import PPOAgent, RandomAgent
from rlcard.utils import set_seed, reorganize


def plot_learning_curve(x, scores, n_games, log_dir, figure_file):
    # Ensure the directory exists before saving the figure
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title(f"Running average of previous {n_games} scores")
    plt.savefig(figure_file)


def train(args):
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        'no-limit-holdem',
        config={
            'seed': args.seed,
        }
    )

    N = 1
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    ppo = PPOAgent(n_actions=env.num_actions, batch_size=batch_size,
                   alpha=alpha, n_epochs=n_epochs,
                   input_dims=env.state_shape[0], save_path=args.log_dir)
    env.set_agents([ppo, RandomAgent(num_actions=env.num_actions)])

    figure_file = 'poker.png'

    best_score = -100
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(args.num_episodes):
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        score = 0
        for ts in trajectories[0]:
            n_steps += 1
            score += ts[2]
            ppo.feed(ts)
            if n_steps % N == 0:
                ppo.train()
                learn_iters += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            ppo.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, args.num_episodes, args.log_dir, args.log_dir + figure_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/nolimitholdem_ppo_result3/',
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    train(args)
