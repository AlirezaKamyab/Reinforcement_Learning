import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DynaQMaze(gym.Env):
    def __init__(self, 
                 rows: int, 
                 columns: int,
                 *,
                 num_blocks: int=None,
                 block_positions: list=None,
                 start_location: tuple=None,
                 target_location: tuple=None, 
                 step_reward:float=0.0,
                 target_reward:float=1.0):
        super(DynaQMaze, self).__init__()

        self.rows = rows
        self.columns = columns
        self.num_blocks = num_blocks
        self.block_positions = block_positions
        self.target_location = target_location
        self.start_location = start_location
        self.target_reward = target_reward
        self.step_reward = step_reward

        self.observation = None

        assert block_positions is None or num_blocks == len(block_positions) or num_blocks is None, "Inconsistant number of blocks"
        self.num_blocks = len(block_positions)

        self.observation_space = gym.spaces.Dict({
            'agent': gym.spaces.Box(low=np.array([0, 0]), high=np.array([rows - 1, columns - 1]), shape=(2, ), dtype=int),
            'target': gym.spaces.Box(low=np.array([0, 0]), high=np.array([rows - 1, columns - 1]), shape=(2, ), dtype=int),
            'block_positions': gym.spaces.Box(low=np.array([[0, 0]] * self.num_blocks), 
                                              high=np.array([[rows - 1, columns - 1]] * self.num_blocks), 
                                              shape=(self.num_blocks, 2), dtype=int)
        })

        self.action_space = gym.spaces.Discrete(4)
        self.action_to_direction = {
            0: np.array([1, 0]), # Down
            1: np.array([0, 1]), # right
            2: np.array([-1, 0]), # Up
            3: np.array([0, -1]), # left
        }


    def is_terminated(self):
        assert self.observation is not None, "Please reset the environment first!"
        return bool(np.all(self.observation['agent'] == self.observation['target']))


    def reset(self):
        obs = self.observation_space.sample()
        if self.start_location is not None:
            obs['agent'] = np.array(self.start_location, dtype=np.int32)
        if self.block_positions is not None:
            obs['block_positions'] = np.array(self.block_positions, dtype=np.int32)
        if self.target_location is not None:
            obs['target'] = np.array(self.target_location, dtype=np.int32)
        
        self.observation = obs
        return obs['agent']
    

    def step(self, action:int):
        assert self.observation is not None, "Please reset the environment first!"
        if self.is_terminated():
            return self.observation, 0, True

        state = self.observation['agent']
        target = self.observation['target']
        blocks = self.observation['block_positions']

        direction = self.action_to_direction[action]
        new_state = state + direction

        hit_block = False
        for b in range(blocks.shape[0]):
            if(np.all(blocks[b] == new_state)):
                hit_block = True
                break

        if hit_block:
            new_state = state
        elif new_state[0] < 0 or new_state[1] < 0:
            new_state = state
        elif new_state[0] >= self.rows or new_state[1] >= self.columns:
            new_state = state

        # Update the state
        self.observation['agent'] = new_state
        
        terminated = self.is_terminated()

        reward = self.step_reward
        if np.all(new_state == target):
            reward = self.target_reward

        return self.observation['agent'], reward, terminated, None, None

        
def draw_maze(env: DynaQMaze):
    assert env.observation is not None, "Reset the environment first!"
    obs = env.observation

    # Grid size
    rows, cols = env.rows, env.columns

    # Some sample coordinates to fill with color
    black_cells = obs['block_positions'].tolist()
    green_cells = [obs['agent'].tolist()]
    red_cells = [obs['target'].tolist()]

    # Create the figure and axes
    ax = plt.subplot(111)

    # Draw the grid
    for x in range(cols):
        for y in range(rows):
            rect = patches.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)

    # Fill selected squares with color
    for y, x in black_cells:
        rect = patches.Rectangle((x, y), 1, 1, facecolor='black')
        ax.add_patch(rect)

    for y, x in green_cells:
        rect = patches.Rectangle((x, y), 1, 1, facecolor='green')
        ax.add_patch(rect)

    for y, x in red_cells:
        rect = patches.Rectangle((x, y), 1, 1, facecolor='red')
        ax.add_patch(rect)

    # Set limits and aspect
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axis if you want just the grid

    plt.gca().invert_yaxis()  # Optional: flip Y-axis to have (0,0) at top-left like a matrix
    plt.show(block=False)


if __name__ == "__main__":
    start_location = (2, 0)
    target_location = (0, 8)
    rows = 6
    columns = 9
    block_positions = [[1, 2], [2, 2], [3, 2], [4, 5], [0, 7], [1, 7], [2, 7]]
    env = DynaQMaze(rows=rows, 
                    columns=columns, 
                    block_positions=block_positions, 
                    start_location=start_location, 
                    target_location=target_location)
    print(env.reset())
    plt.figure()
    while True:
        draw_maze(env)
        action = int(input(">> "))
        print(env.step(action=action))
        draw_maze(env)
