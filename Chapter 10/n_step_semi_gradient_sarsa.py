import numpy as np
from tile_coding import TileCoder
import gymnasium as gym
from tqdm import tqdm


def choose_action(w:np.ndarray, x:np.ndarray, epsilon:float=0.0):
    num_actions = w.shape[-1]
    if np.random.rand() < epsilon:
        return int(np.random.choice(num_actions))
    
    values = w.T @ x
    max_v = np.max(values)
    ivalues = np.argwhere(values == max_v)[:, 0]
    return int(np.random.choice(ivalues.tolist()))


def n_step_semi_gradient_sarsa(
    episodes:int,
    tile_coder:TileCoder,
    env:gym.Env,
    num_actions:int,
    n:int=1,
    gamma:float=1.0,
    alpha:float=0.1,
    epsilon:float=0.1
):
    w = np.zeros((tile_coder.num_features, num_actions), dtype=np.float32)
    memory = [{} for _ in range(n + 1)]
    # history
    steps_per_episode = np.zeros(episodes, dtype=np.int32)

    for episode in tqdm(range(episodes), desc="Training on episode", leave=False):
        T = np.inf
        t = 0

        state = env.reset()[0]
        feature = tile_coder.get_vector(state)
        action = choose_action(w, feature, epsilon=epsilon)
        # save to memory
        memory[t % (n + 1)]['state'] = state
        memory[t % (n + 1)]['action'] = action
        
        while True:
            if t < T:
                state = memory[t % (n + 1)]['state']
                action = memory[t % (n + 1)]['action']
                # take a step and observe
                new_state, reward, terminated, _, _ = env.step(action)

                # save to memory
                memory[(t + 1) % (n + 1)]['state'] = new_state
                memory[(t + 1) % (n + 1)]['reward'] = reward

                if terminated:
                    T = t + 1
                else:
                    # If the game is not over take another action and save
                    new_feature = tile_coder.get_vector(new_state)
                    new_action = choose_action(w, new_feature, epsilon=epsilon)
                    memory[(t + 1) % (n + 1)]['action'] = new_action

            tau = t + 1 - n
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n + 1, T + 1)):
                    G += np.power(gamma, tau - i - 1) * memory[i % (n + 1)]['reward']

                if tau + n < T:
                    n_state = memory[(tau + n) % (n + 1)]['state']
                    n_action = memory[(tau + n) % (n + 1)]['action']
                    n_feature = tile_coder.get_vector(n_state)
                    n_value = w.T @ n_feature
                    G += np.power(gamma, n) * n_value[n_action]

                tau_state = memory[tau % (n + 1)]['state']
                tau_feature = tile_coder.get_vector(tau_state)
                tau_action = memory[tau % (n + 1)]['action']
                tau_value = w.T @ tau_feature

                # Update the parameters
                w[:, tau_action] += alpha * (G - tau_value[tau_action]) * tau_feature
            
            t += 1
            if tau == T + 1:
                steps_per_episode[episode] = T
                break
    return w, steps_per_episode
                

# Implementation based on SAB
coder = TileCoder(
    low=[-1.2, -0.07],
    high=[0.6, 0.07],
    num_tiles=8,
    tile_per_dimension=[8, 8]
)

num_actions = 3
env = gym.make('MountainCar-v0', render_mode='rgb_array')

w, history = n_step_semi_gradient_sarsa(
    episodes=500, 
    tile_coder=coder,
    env=env,
    num_actions=num_actions,
    n=3,
    alpha=0.2/8
)

print(w)
print(history)