import os
import time
import argparse
from environment import Environment
from sarsa_agent import SarsaAgent
from qlearn_agent import QLearnAgent

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_episodes', type=int, default=500, help='Number of training episodes', metavar='')
  parser.add_argument('--n_rows', type=int, default=13, help='Number of rows in the labyrinth', metavar='')
  parser.add_argument('--n_cols', type=int, default=33, help='Number of cols in the labyrinth', metavar='')
  parser.add_argument('--n_walls', type=int, default=100, help='Number of walls in the labyrinth', metavar='')
  parser.add_argument('--epsilon', type=float, default=0.01, help='Agent epsilon for eps-greedy policy', metavar='')
  parser.add_argument('--alpha', type=float, default=1.0, help='Agent alpha for step size', metavar='')
  parser.add_argument('--gamma', type=float, default=1.0, help='Return discount factor', metavar='')
  parser.add_argument('--mode', choices=['qlearn', 'sarsa'], default='qlearn')

  return parser.parse_args()


if __name__ == '__main__':

  # Parse command line arguments
  args = parse_arguments()

  environment = Environment(n_rows=args.n_rows, n_cols=args.n_cols, n_walls=args.n_walls)

  if args.mode == 'qlearn':
    agent = QLearnAgent(n_rows=args.n_rows, n_cols=args.n_cols, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma)
  elif args.mode == 'sarsa':
    agent = SarsaAgent(n_rows=args.n_rows, n_cols=args.n_cols, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma)

  for e in range(args.n_episodes):

    cumulative_reward = 0
    state = environment.start_new_episode()

    if args.mode == 'qlearn':
      # Classic reinforcement learning loop
      is_over = False
      while True:

        # Choose agent's action (epsilon greedy policy)
        action = agent.get_action_eps_greedy(*state)

        # Perform action in the environment
        new_state, reward, is_over = environment.perform_action(action)

        # Update agent's Q function
        agent.update_Q(state, action, reward, new_state)

        # Update cumulative reward
        cumulative_reward += reward

        # Update state
        state = new_state

        if is_over:
          break
    
    elif args.mode == 'sarsa':

      # Get initial action
      action = agent.get_action_eps_greedy(*state)

      is_over = False
      while True:

        # Perform action in the environment
        new_state, reward, is_over = environment.perform_action(action)

        # Update agent's Q function
        next_action = agent.update_Q(state, action, reward, new_state)

        # Update cumulative reward
        cumulative_reward += reward

        # Update state and action
        state = new_state
        action = next_action

        if is_over:
          break

    print('Episode: {:03d} - Cumulative reward this episode: {}'.format(e, cumulative_reward))

  input('End of training. \n\nPress `ENTER` to start testing.')

  state = environment.start_new_episode()
  is_over = False
  while True:

    os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
    print('Learnt policy:')
    print(environment.policy_str(agent))
    print('Testing policy:')
    print(environment)

    # Choose agent's action (greedy policy)
    action = agent.get_action_greedy(*state)

    # Perform action in the environment
    new_state, reward, is_over = environment.perform_action(action)

    # Update state
    state = new_state

    if is_over:
      break

    time.sleep(0.5)
