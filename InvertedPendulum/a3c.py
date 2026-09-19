import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
from typing import Tuple


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0.0)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim:int, action_dim:int):
        super(ActorNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)
        )

        for layer in self.net:
            if not hasattr(layer, 'weight'):
                continue
            layer.weight.data.mul_(0.01)
            layer.bias.data.fill_(0.0)

        # self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        mu, log_sigma = torch.chunk(x, 2, dim=-1)
        # sigma = torch.exp(self.log_std).expand_as(mu)
        sigma = torch.exp(log_sigma)
        return mu, sigma


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim:int):
        super(CriticNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)


def worker(
    worker_id:int,
    env_name:str,
    global_actor_network:nn.Module,
    global_critic_network:nn.Module,
    optimizer_actor:torch.optim.Optimizer,
    optimizer_critic:torch.optim.Optimizer,
    gamma:float,
    episodes:int,
    t_max:int=1
):
    torch.set_num_threads(1)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    local_actor = ActorNetwork(obs_dim, action_dim)
    local_critic = CriticNetwork(obs_dim)

    local_actor.load_state_dict(global_actor_network.state_dict())
    local_critic.load_state_dict(global_critic_network.state_dict())

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        done = False
        episode_rewards = 0.0

        while not done:
            log_probs, values, rewards = [], [], []

            for _ in range(t_max):
                mu, sigma = local_actor(state)
                dist = torch.distributions.Normal(mu, sigma)
                action = dist.sample()

                log_prob = dist.log_prob(action).sum(dim=-1)
                value = local_critic(state)

                np_action = action.detach().numpy().clip(action_low, action_high)
                next_state, reward, terminated, truncated, _ = env.step(np_action)
                episode_rewards += reward
                next_state = torch.tensor(next_state, dtype=torch.float32)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)

                state = next_state

                if done:
                    break

            R = 0.0 if terminated else local_critic(state).item()
            returns = []

            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
            values = torch.stack(values)
            log_probs = torch.stack(log_probs).unsqueeze(1)

            advantages = returns - values

            # actor and critic losses
            critic_loss = (returns - values).square().mean()
            actor_loss = -(advantages.detach() * log_probs).mean()

            # global updates
            optimizer_critic.zero_grad()
            local_critic.zero_grad()
            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(local_critic.parameters(), 10)

            for local_param, global_param in zip(local_critic.parameters(), global_critic_network.parameters()):
                global_param.grad = local_param.grad
            optimizer_critic.step()


            optimizer_actor.zero_grad()
            local_actor.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(local_actor.parameters(), 10)

            for local_param, global_param in zip(local_actor.parameters(), global_actor_network.parameters()):
                global_param.grad = local_param.grad
            optimizer_actor.step()

            local_actor.load_state_dict(global_actor_network.state_dict())
            local_critic.load_state_dict(global_critic_network.state_dict())
        if episode % 50 == 0:
            print(f"Worker {worker_id} | Episode: {episode:4d} | Reward: {episode_rewards:6.2f}")

    env.close()


def main():
    env_name = "InvertedPendulum-v5"
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    global_actor = ActorNetwork(obs_dim, action_dim)
    global_critic = CriticNetwork(obs_dim)

    global_actor.share_memory()
    global_critic.share_memory()

    optimizer_actor = SharedAdam(global_actor.parameters(), lr=1e-4)
    optimizer_critic = SharedAdam(global_critic.parameters(), lr=5e-4)

    processes = []
    gamma = 0.99
    num_workers = 8
    episodes = 200
    t_max = 20


    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                worker_id,
                env_name,
                global_actor,
                global_critic,
                optimizer_actor,
                optimizer_critic,
                gamma,
                episodes,
                t_max
            )
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return global_actor


@torch.no_grad()
def evaluation_play(actor_network:nn.Module):
    env = gym.make("InvertedPendulum-v5", render_mode='human')
    env.metadata["render_fps"] = 1
    terminated = False
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32)
    total_reward = 0
    while not terminated:
        mu, sigma = actor_network(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample().detach().numpy().clip(-3, 3)
        state, reward, terminated, _, _ = env.step(
            action
        )
        state = torch.tensor(state, dtype=torch.float32)
        total_reward += reward

    env.close()
    return total_reward


if __name__ == '__main__':
    mp.set_start_method('spawn')

    actor_network = main()
    evaluation_play(actor_network)