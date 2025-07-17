import gymnasium as gym
import numpy as np
from typing import Optional

class WindyGridWorld(gym.Env):
    def __init__(self, 
                 rows: int, 
                 columns: int, 
                 *, 
                 init_location: np.ndarray= None, 
                 king_move: bool=False):
        super(WindyGridWorld, self).__init__()

        self.rows = rows
        self.columns = columns

        self.init_location = init_location
        self.current_location = None
        self.target_location = None
        self.wind_location = None

        self.observation_space = gym.spaces.Dict(
            {
                'agent': gym.spaces.Box(low=np.array([0, 0]), high=np.array([rows, columns]), shape=(2, ), dtype=int),
                'target': gym.spaces.Box(low=np.array([0, 0]), high=np.array([rows, columns]), shape=(2, ), dtype=int),
                'wind_location': gym.spaces.Box(low=0, high=columns // 2, shape=(rows, ), dtype=int)
            }
        )

        if not king_move:
            self.action_space = gym.spaces.Discrete(4)
        else:
            self.action_space = gym.spaces.Discrete(8)

        if not king_move:
            self.action_to_direction = {
                0: np.array([1, 0]), # down
                1: np.array([0, 1]), # right
                2: np.array([-1, 0]), # up
                3: np.array([0, -1]) # left
            }
        else:
            self.action_to_direction = {
                0: np.array([1, 0]), # down
                1: np.array([0, 1]), # right
                2: np.array([-1, 0]), # up
                3: np.array([0, -1]), # left
                4: np.array([1, 1]), # down-right
                5: np.array([1, -1]), # down-left
                6: np.array([-1, 1]), # up-right
                7: np.array([-1, -1]), # up-left
            }

    def _get_obs(self):
        return {'agent': self.current_location, 'target': self.target_location}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        random_observation = self.observation_space.sample()
        if self.init_location is None:
            self.current_location = random_observation['agent']
        else:
            self.current_location = self.init_location
        if self.target_location is None:
            self.target_location = random_observation['target']
        if self.wind_location is None:
            self.wind_location = random_observation['wind_location']

        return self._get_obs()
    
    def is_terminated(self):
        return bool(np.all(self.current_location == self.target_location))

    def step(self, action):
        direction = self.action_to_direction[action]
        self.current_location = np.clip(self.current_location + direction, [0, 0], [self.rows - 1, self.columns - 1])

        # Compute the wind power
        current_column = self.current_location[1]
        wind_power = np.array([self.wind_location[current_column], 0], dtype=np.int32)
        # apply wind power
        self.current_location = np.clip(self.current_location + wind_power,
                                        [0, 0],
                                        [self.rows - 1, self.columns - 1])
        terminated = bool(np.all(self.current_location == self.target_location))
        truncated = False
        reward = 1 if terminated else -1
        observation = self._get_obs()
        info = None

        return observation, reward, terminated, truncated, info
