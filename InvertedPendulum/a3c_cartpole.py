import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.multiprocessing as mp


class Q_network(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super(Q_network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def epsilon_greedy(values:torch.Tensor, epsilon:float):
    if np.random.rand() < epsilon:
        return np.random.choice(values.shape[0])

    action = torch.argmax(values)
    return action


def worker(
    worker_id:int,
    global_model:nn.Module,
    optimizer:torch.optim.Optimizer,
    env_name:str,
    gamma:float,
    epsilon:float,
    episodes:int
):
    torch.set_num_threads(1)
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    local_model = Q_network(obs_dim, action_dim)

    for episode in range(episodes):
        # synchronize the local model with the global model
        local_model.load_state_dict(global_model.state_dict())

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        episode_rewards = 0.0

        while not done:
            state_values = local_model(state)
            action = epsilon_greedy(state_values, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            episode_rewards += reward
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)

            if terminated:
                td_target = reward
            else:
                with torch.no_grad():
                    next_state_values = local_model(next_state)
                    max_value = torch.max(next_state_values, dim=-1).values
                    td_target = reward + gamma * max_value

            td_error = td_target - state_values[int(action)]
            loss = 0.5 * (td_error ** 2)

            optimizer.zero_grad()
            local_model.zero_grad()

            loss.backward()

            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                if local_param is not None:
                    global_param._grad = local_param._grad

            optimizer.step()

            # sync after each update
            local_model.load_state_dict(global_model.state_dict())

            state = next_state

        if episode % 10 == 0:
            print(
                f"Worker {worker_id} | "
                f"Episode {episode} | "
                f"Reward {episode_rewards:.1f}"
            )

    env.close()


def main():
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    env.close()

    # instaniate the global model
    global_model = Q_network(obs_dim, action_dim)
    global_model.share_memory()

    optimizer = torch.optim.SGD(global_model.parameters(), lr=1e-4)

    gamma = 0.99
    epsilon = 0.1

    num_workers = 8
    episodes = 1000

    processes = []

    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                worker_id,
                global_model,
                optimizer,
                env_name,
                gamma,
                epsilon,
                episodes
            )
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()