#!/usr/bin/env python3
from typing import Tuple


class ShortCorridorEnv:
    def __init__(
            self, 
            num_states:int=4,
            inverted:list=[False, True, False, False],
            truncate:int=None
        ):
        self.state = None
        self.num_states = num_states
        self.inverted = inverted
        self.terminal_state = num_states - 1
        self.action_to_direction = [1, -1]
        self.terminated = False
        self.truncate = truncate

    def reset(self) -> int:
        self.state = 0
        self.terminated = False
        self.steps = 0
        return self.state

    def step(self, action:int) -> Tuple[float, int, bool]:
        assert self.state is not None, "Reset the environment first!"
        assert action in [0, 1], "Invalid action is taken!"
        if self.terminated:
            return 0, self.state, self.terminated

        direction = self.action_to_direction[action]
        if self.inverted[self.state]:
            direction = direction * -1

        self.state = min(max(self.state + direction, 0), self.terminal_state)
        reward = -1
        if self.state == self.terminal_state:
            self.terminated = True

        self.steps += 1
        if self.truncate is not None and self.steps >= self.truncate:
            self.terminated = True

        return reward, self.state, self.terminated


if __name__ == '__main__':
    import numpy as np

    sce = ShortCorridorEnv()
    sce.reset()
    print(f"State: {sce.state}")
    while not sce.terminated:
        action = np.random.choice(2)
        r, s, t = sce.step(action)
        print(f"Action: {action}")
        print(f"State: {s}, Reward: {r}, Term: {t}")

