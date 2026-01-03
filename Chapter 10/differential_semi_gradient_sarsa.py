from tile_coding import TileCoder
import numpy as np
import gymnasium as gym
from tqdm import tqdm


def choose_action(w:np.ndarray, x:np.ndarray, epsilon:float=0.0):
    num_actions = w.shape[-1]
    if np.random.rand() < epsilon:
        return int(np.random.choice(num_actions))
    
    values = w.T @ x
    max_q = np.max(values)
    candidates = np.argwhere(values == max_q)[:, 0]
    return int(np.random.choice(candidates.tolist()))


def differential_semi_gradient_sarsa(
    episodes:int,
    tilecoder:TileCoder,
    env:gym.Env,
    num_actions:int,
    epsilon:float=0.1,
    alpha:float=0.1,
    beta:float=0.1
):
    w = np.zeros((tilecoder.num_features, num_actions), dtype=np.float32)
    steps_per_episode = np.zeros(episodes, dtype=np.int32)
    R_bar = 0
    for episode in tqdm(range(episodes), desc="Training episode", leave=False):
        state = env.reset()[0]
        feature = tilecoder.get_vector(state)
        action = choose_action(w, feature, epsilon=epsilon)
        terminated = False
        
        while not terminated:
            # Observe S' and R by taking action
            next_state, reward, terminated, _, _ = env.step(action)
            next_feature = tilecoder.get_vector(next_state)
            next_action = choose_action(w, next_feature, epsilon=epsilon)

            next_value = (w.T @ next_feature)[next_action]
            current_value = (w.T @ feature)[action]

            td_error = reward - R_bar + next_value - current_value
            w[:, action] = w[:, action] + alpha * td_error * feature
            R_bar = R_bar + beta * td_error

            state = next_state
            feature = next_feature
            action = next_action
            steps_per_episode[episode] += 1
    
    return w, steps_per_episode, R_bar


if __name__ == '__main__':
    # Implementation based on SAB
    coder = TileCoder(
        low=[-1.2, -0.07],
        high=[0.6, 0.07],
        num_tiles=8,
        tile_per_dimension=[8, 8]
    )

    num_actions = 3
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    w, history, R_bar = differential_semi_gradient_sarsa(
        episodes=500, 
        tilecoder=coder, 
        env=env,
        num_actions=num_actions,
        epsilon=0.1,
        alpha=0.1/8,
        beta=0.01,
    )

    print(history)
    print(R_bar)