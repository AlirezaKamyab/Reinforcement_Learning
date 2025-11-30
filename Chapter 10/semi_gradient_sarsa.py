import gymnasium as gym
import numpy as np
from tile_coding import TileCoder
from tqdm import tqdm


def epsilon_greedy_policy(
    epsilon:float=0.0,
    *,
    values:np.ndarray=None,
    x:np.ndarray=None,
    w:np.ndarray=None
):
    if values is None and (x is not None and w is not None):
        values = w.T @ x
    elif values is None and (x is None or w is None):
        raise ValueError("You should give either values or x and w")
    
    if np.random.rand() < epsilon:
        return np.random.choice(values.shape[0])
    return np.argmax(values)


def semi_gradient_sarsa(
    env:gym.Env,
    tile_coder:TileCoder,
    num_actions:int,
    epsilon:float=0.1,
    episodes:int=1,
    alpha:float=0.1,
    gamma:float=1.0
):
    w = np.zeros(shape=(tile_coder.num_features, num_actions), dtype=np.float32)
    steps_per_episode = np.zeros(episodes, dtype=np.int32)

    with tqdm(range(episodes), leave=False) as p_bar:
        for episode in p_bar:
            state = env.reset()[0]
            x = tile_coder.get_vector(state)
            action = epsilon_greedy_policy(epsilon=epsilon, x=x, w=w)
            terminated = False
            while not terminated:
                new_state, reward, terminated, truncated, _ = env.step(action)
                x_prime = tile_coder.get_vector(new_state)
                # Computes values for each action in the new state
                values = w.T @ x_prime
                new_action = epsilon_greedy_policy(epsilon=epsilon, values=values)

                # Computes values for each action in the old stae
                values_old = w.T @ x

                # Update the weights
                if not terminated:
                    w[:, action] = w[:, action] + alpha * (reward + gamma * values[new_action] - values_old[action]) * x
                else:
                    w[:, action] = w[:, action] + alpha * (reward - values_old[action]) * x
                    continue

                # Update for next step
                state = new_state
                action = new_action
                x = tile_coder.get_vector(state)
                steps_per_episode[episode] += 1
            
            # End of the episode
            p_bar.set_postfix({'total_steps':steps_per_episode[episode]})

    return w, steps_per_episode