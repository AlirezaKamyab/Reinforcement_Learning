import numpy as np
import gymnasium as gym
from collections import defaultdict


class DynaQ:
    def __init__(self, 
                 *, 
                 num_actions:int,
                 add_bonus:bool=False,
                 planning_steps:int=0,
                 gamma: float=1.0,
                 alpha: float=0.1,
                 epsilon: float=0.1,
                 kappa: float=0.01, 
                 seed:int=None):
    
        self.num_actions = num_actions
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.model = {}
        self.last_visited = defaultdict(lambda: np.zeros(num_actions))
        self.N = defaultdict(lambda: np.zeros(num_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.kappa = kappa
        self.add_bonus = add_bonus  
        self.time_step = 0
        self.planning_steps = planning_steps

        self.steps_per_episode = []
        self.cumulative_rewards_per_step = [0]

        if seed is not None:
            np.random.seed(seed)

    def choose_action(self, state):
        """
        Chooses actions according to epsilon-greedy policy
        """
        state = tuple(state)
        num_actions = self.num_actions
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(num_actions))
        
        max_value = np.max(self.Q[state])
        candidates = np.flatnonzero(self.Q[state] == max_value)
        chosen = int(np.random.choice(candidates))
        return chosen
    
    def one_step_TD_update(self, state:np.ndarray, action:int, new_state:np.ndarray, reward:float):
        """
        Applies one-step TD udpate
        """
        state, new_state = tuple(state), tuple(new_state)
        max_q = np.max(self.Q[new_state])
        td_target = reward + self.gamma * max_q - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_target

    def update_model(self, state:np.ndarray, action:int, new_state:np.ndarray, reward:float):
        """
        Learns the model
        """
        state, new_state = tuple([int(x) for x in state]), tuple([int(x) for x in new_state])
        self.model[state, action] = (new_state, reward)
        self.N[state][action] += 1
        self.last_visited[state][action] = self.time_step

    def one_step_planning(self, state, action):
        """
        Learns from one-step simulated experience
        """
        new_state, reward = self.model[state, action]

        bonus = 0
        if self.add_bonus:
            tau = self.time_step - self.last_visited[state][action]
            bonus = self.kappa * np.sqrt(tau)

        self.one_step_TD_update(state, action, new_state, reward + bonus)


    def plan(self):
        """
        Learns from simulated experience
        """
        if not self.model:
            return
        keys = list(self.model.keys())
        for _ in range(self.planning_steps):
            if len(keys) == 0: return
            idx = np.random.choice(len(keys))
            state, action = keys.pop(idx)
            self.one_step_planning(state, action)

    
    def run_episode(self, env:gym.Env, max_steps:int=None):
        state = env.reset()
        steps = 0
        while True:
            new_state, _, terminated = self.step(state, env)
            state = new_state
            steps += 1
            if terminated: 
                self.steps_per_episode.append(steps)
                break

            if max_steps is not None and steps >= max_steps:
                self.steps_per_episode.append(steps)
                break

    def step(self, state, env:gym.Env):
        action = self.choose_action(state)
        new_state, reward, terminated, _, _ = env.step(action)
        self.cumulative_rewards_per_step.append(self.cumulative_rewards_per_step[-1] + reward)

        self.one_step_TD_update(state, action, new_state, reward)
        self.update_model(state, action, new_state, reward)
        self.plan()
        self.time_step += 1
        return new_state, reward, terminated

    
    def get_policy(self):
        policy = {}
        for k, v in self.Q.items():
            policy[k] = int(np.argmax(v))
        
        return policy
