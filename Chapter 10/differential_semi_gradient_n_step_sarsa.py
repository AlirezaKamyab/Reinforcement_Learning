from tile_coding import TileCoder
import numpy as np
from tqdm import tqdm
import gymnasium as gym


def choose_action(w:np.ndarray, x:np.ndarray, epsilon:float=0.0):
    num_actions = w.shape[-1]
    if np.random.rand() < epsilon:
        return int(np.random.choice(num_actions))
    
    values = w.T @ x
    max_q = np.max(values)
    candidates = np.argwhere(values == max_q)[:, 0]
    return int(np.random.choice(candidates.tolist()))


def differential_semi_gradient_n_step_sarsa(
    episodes:int,
    tilecoder:TileCoder,
    env: gym.Env,
    num_actions:int,
    n:int=1,
    epsilon:float=0.1,
    alpha:float=0.1,
    beta:float=0.1,
):
    w = np.zeros((tilecoder.num_features, num_actions), dtype=np.float32)
    R_bar = 0.0
    memory = [{} for _ in range(n + 1)]
    steps_per_episode = np.zeros(episodes)

    with tqdm(range(episodes), desc="Training episode", leave=False) as p_bar:
        for episode in p_bar:
            state = env.reset()[0]
            feature = tilecoder.get_vector(state)
            action = choose_action(w, feature, epsilon=epsilon)

            t = 0
            T = np.inf
            memory[t % (n + 1)]['state'] = state
            memory[t % (n + 1)]['feature'] = feature
            memory[t % (n + 1)]['action'] = action 

            terminated = False

            while True:
                if t < T:
                    state = memory[t % (n + 1)]['state']
                    feature = memory[t % (n + 1)]['feature']
                    action = memory[t % (n + 1)]['action']

                    # take action and observe
                    new_state, reward, terminated, _, _ = env.step(action)
                    new_feature = tilecoder.get_vector(new_state)
                    
                    # save to memory
                    memory[(t + 1) % (n + 1)]['state'] = new_state
                    memory[(t + 1) % (n + 1)]['feature'] = new_feature
                    memory[(t + 1) % (n + 1)]['reward'] = reward

                    if terminated:
                        T = t + 1
                    else:
                        new_action = choose_action(w, new_feature, epsilon=epsilon)
                        memory[(t + 1) % (n + 1)]['action'] = new_action
                
                tau = t + 1 - n
                if tau >= 0:
                    diff_rewards = 0
                    for i in range(tau + 1, min(T + 1, tau + n + 1)):
                        diff_rewards += memory[i % (n + 1)]['reward'] - R_bar
                    
                    new_feature = memory[(tau + n) % (n + 1)]['feature']
                    new_action = memory[(tau + n) % (n + 1)]['action']
                    new_q = (w.T @ new_feature)[new_action]

                    feature = memory[tau % (n + 1)]['feature']
                    action = memory[tau % (n + 1)]['action']
                    q = (w.T @ feature)[action]

                    td_error = diff_rewards + new_q - q
                    w[:, action] += alpha * td_error * feature
                    R_bar = R_bar + beta * td_error

                t += 1
                if tau == T - 1: break
                
            p_bar.set_postfix({'average_reward':R_bar, 'steps':t})
            steps_per_episode[episode] = t
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
    w, history, R_bar = differential_semi_gradient_n_step_sarsa(
        episodes=500, 
        tilecoder=coder, 
        env=env,
        num_actions=num_actions,
        n=4,
        epsilon=0.1,
        alpha=0.1/8,
        beta=0.01,
    )

    print(history)
    print(R_bar)