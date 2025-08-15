import numpy as np
import heapq
from collections import defaultdict
import gymnasium as gym

class PrioritizedSweeping:
    def __init__(
        self,
        num_actions:int,
        gamma:float=1.0,
        alpha:float=0.1,
        epsilon:float=0.1,
        planning_steps:int=1,
        theta:float=0.01,
        seed:int=None,
        max_buffer:int=None
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.theta = theta
        self.max_buffer = max_buffer

        self.Q = defaultdict(lambda: np.zeros(shape=(num_actions,), dtype=np.float32))
        self.model = {}
        self.priority_queue = []

        if seed is not None:
            np.random.seed(seed)

    @classmethod
    def convert_to_tuple(cls, state):
        """
        If state is in ndarray, we convert it to a tuple
        """
        state = tuple([int(x) for x in state])
        return state

    def choose_action(self, state):
        """
        With the probability of epsilon, we take an action randomly
        With the probability of 1-epsilon, we take a greedy action (ties broken randomly)
        """
        state = self.convert_to_tuple(state)
        if np.random.rand() < self.epsilon:
            chosen = np.random.choice(self.num_actions)
            return int(chosen)
        else:
            max_q = np.max(self.Q[state])
            candidates = np.flatnonzero(self.Q[state] == max_q)
            chosen = np.random.choice(candidates)
            return int(chosen)
        
    def one_step_TD_update(
            self, 
            state:np.ndarray, 
            action:int, 
            new_state:np.ndarray, 
            reward:float, 
            add_to_queue:bool=True
        ):
        """
        Applies direct reinforcement learning update with one-step TD update
        """
        state, new_state = self.convert_to_tuple(state), self.convert_to_tuple(new_state)
        action = int(action)
        max_q = np.max(self.Q[new_state])
        td_target = reward + self.gamma * max_q - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_target
        if add_to_queue and np.abs(td_target) >= self.theta:
            heapq.heappush(self.priority_queue, (-np.abs(td_target), state, action))

    def update_model(
            self, 
            state:np.ndarray, 
            action:np.ndarray, 
            new_state:np.ndarray, 
            reward:float):
        """
        Learns the model by assuming that the model is deterministic
        """
        state, new_state = self.convert_to_tuple(state), self.convert_to_tuple(new_state)
        action = int(action)
        self.model[state, action] = (new_state, reward)

    def one_step_planning(self, state:np.ndarray, action:int):
        """
        Applies one-step simulated update
        """
        state = self.convert_to_tuple(state)
        action = int(action)
        new_state, reward = self.model[state, action]
        self.one_step_TD_update(state, action, new_state, reward, add_to_queue=False)

        for state_bar, action_bar in self.model.keys():
            new_state_bar, reward_bar = self.model[state_bar, action_bar]
            if new_state_bar != state: continue
            max_q = np.max(self.Q[new_state_bar])
            td_target = np.abs(reward_bar + self.gamma * max_q - self.Q[state_bar][action_bar])
            if td_target < self.theta: continue
            heapq.heappush(self.priority_queue, (-td_target, state_bar, action_bar))


    def plan(self):
        """
        Run multiple planning steps
        """
        if not self.model:
            return
        
        for _ in range(self.planning_steps):
            if len(self.priority_queue) == 0: 
                break
            _, state, action = heapq.heappop(self.priority_queue)
            self.one_step_planning(state, action)

    def step(self, state:np.ndarray, env:gym.Env):
        """
        In a single step we apply, direct-ml, update model and learn via simulated experience
        """
        action = self.choose_action(state)
        new_state, reward, terminated, _, _ = env.step(action)
        self.one_step_TD_update(state, action, new_state, reward)
        self.update_model(state, action, new_state, reward)
        self.plan()
        if self.max_buffer and len(self.priority_queue) > 0 and len(self.priority_queue) > self.max_buffer:
            self.priority_queue = heapq.nsmallest(self.max_buffer, self.priority_queue)
            heapq.heapify(self.priority_queue)
        return new_state, reward, terminated
    

    def run_episode(self, env:gym.Env):
        state = env.reset()
        steps = 0
        while True:
            state, _, terminated = self.step(state, env)
            steps += 1
            if terminated: break
        
        return {
            "steps": steps,
        }
    
    def run(self, episodes:int, env:gym.Env):
        steps_per_episode = []
        for _ in range(episodes):
            history = self.run_episode(env)
            steps_per_episode.append(history['steps'])
        return {
            'steps':steps_per_episode,
        }

