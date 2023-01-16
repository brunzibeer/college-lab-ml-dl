from agent import Agent

class SarsaAgent(Agent):
  """
  Class that models a reinforcement learning agent.
  """

  def update_Q(self, old_state, action, reward, new_state):
    """
    Update action-value function Q
    
    Parameters
    ----------
    old_state: tuple
      Previous state of the Environment
    action: int
      Action performed to go from `old_state` to `new_state`
    reward: int
      Reward got after action `action`
    new_state: tuple
      Next state of the Environment

    Returns
      The action to be executed next
    -------
    None
    """

    next_action = self.get_action_eps_greedy(new_state[0], new_state[1])
    self.Q[old_state[0], old_state[1], action] =\
      self.Q[old_state[0], old_state[1], action] +\
      self.alpha * (reward + self.gamma * self.Q[new_state[0], new_state[1], next_action] -\
      self.Q[old_state[0], old_state[1], action])
    return next_action