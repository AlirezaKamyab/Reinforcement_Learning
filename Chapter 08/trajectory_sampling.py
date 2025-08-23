import numpy as np
from collections import defaultdict
import gymnasium as gym

class TrajectorySampling:
    def __init__(
        self,
        num_actions:int,
        gamma:float=1.0, 
        alpha:float=0.1,
        epsilon:float=0.1,
        depth:int=None
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.depth = depth

        self.Q = defaultdict(lambda: np.zeros(num_actions, dtype=np.float32))
        self.model = {}


    def _to_tuple(self, state:np.ndarray):
        """
        Convert state:ndarray to state:tuple
        """
        if isinstance(state, np.ndarray):
           state = tuple(state.tolist())
        return state
     
    def choose_action(self, state:np.ndarray):
        """
        Chooses actions according to epsilon-greedy policy
        With probability epsilon, a random action is taken
        With probability 1 - epsilon, a greedy action is taken
        Ties are broken randomly
        """
        state = self._to_tuple(state)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.num_actions))
        
        max_q = np.max(self.Q[state])
        candidates = np.flatnonzero(self.Q[state] == max_q)
        return int(np.random.choice(candidates))
    
    def one_step_td_update(
        self,
        state:np.ndarray,
        action:int,
        new_state:np.ndarray,
        reward:float
    ):
        state = self._to_tuple(state)
        new_state = self._to_tuple(new_state)
        max_q = np.max(self.Q[new_state])
        td_target = reward + self.gamma * max_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
    
    def update_model(
        self,
        state:np.ndarray,
        action:int,
        new_state:np.ndarray,
        reward:float
    ):
        state = self._to_tuple(state)
        new_state = self._to_tuple(new_state)
        self.model[state, action] = (new_state, reward)
    
    def one_step_planning(
        self,
        state:np.ndarray,
        action:np.ndarray
    ):
        if not self.model:
            return
        
        state = self._to_tuple(state)
        new_state, reward = self.model[state, action]
        new_action = self.choose_action(state)
        td_target = reward + self.gamma * self.Q[new_state][new_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
        return new_state, reward

    def planning(self):
        keys = list(self.model.keys())
        idx = np.random.choice(len(keys))
        state, action = keys[idx]
        depth = 0
        while self.depth is None or depth <= self.depth:
            if (state, action) not in self.model: break
            state, action = self.one_step_planning(state, action)

    
    def step(self, state:np.ndarray, action:int, env:gym.Env):
        new_state, reward, terminated, _, _ = env.step(action)
        self.one_step_td_update(state, action, new_state, reward)
        self.update_model(state, action, new_state, reward)
        self.planning()
        return new_state, reward, terminated
    
    def run_episode(self, env:gym.Env):
        state = env.reset()
        rewards = []
        while True:
            action = self.choose_action(state)
            state, reward, terminated = self.step(state, action, env)
            rewards.append(reward)
            if terminated:
                break
        return rewards
