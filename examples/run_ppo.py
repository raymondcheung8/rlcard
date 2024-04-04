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
import rlcard
import argparse
from rlcard.agents import PPOAgent, RandomAgent, EquityAgent
import torch
from rlcard.utils import (
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)


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

    batch_size = 64
    learning_rate = 0.0003

    ppo = PPOAgent(n_actions=env.num_actions, batch_size=batch_size, alpha=learning_rate,
                   input_dims=env.state_shape[0], save_path=args.log_dir)

    env.set_agents([ppo, EquityAgent(num_actions=env.num_actions)])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            for ts in trajectories[0]:
                ppo.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode + logger.episode_count,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )
        logger.inc_episode_count(args.num_episodes)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(ppo, save_path)
    print('Model saved in', save_path)


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
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='ppo',
        choices=[
            'dqn',
            'nfsp',
            'ppo'
        ],
    )
    args = parser.parse_args()

    train(args)
