import numpy as np
from tile_coding import TileCoder
import gymnasium as gym
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_features(
    state:np.ndarray, coder:TileCoder
):
    state = np.clip(state, coder.low, coder.high)
    x = coder.get_vector(state)
    return x


def pi(theta:np.ndarray, state:np.ndarray, action:np.ndarray) -> np.ndarray:
    mu, sigma = get_mu_sigma(
        theta=theta,
        x=state
    )
    eps = 1e-9
    denom = 1 / (sigma * np.sqrt(2 * np.pi) + eps)
    exp = np.exp(-np.square((action - mu) / (sigma + eps)) / 2)
    return exp * denom


def get_mu_sigma(theta:np.ndarray, x:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta_mu, theta_sigma = np.split(theta, 2)
    mu = x.T@theta_mu
    sigma = np.exp(x.T@theta_sigma)
    return mu, sigma


def grad_ln_pi(theta:np.ndarray, state:np.ndarray, action:float) -> Tuple[np.ndarray, np.ndarray]:
    mu, sigma = get_mu_sigma(
        theta=theta,
        x=state
    )

    eps = 1e-9
    variance = np.square(sigma)
    grad_mu = np.einsum(
        'd, a->da',
        state,
        (1 / (variance + eps)) * (action - mu)
    )

    grad_sigma = np.einsum(
        'd, a->da',
        state,
        (np.square(action - mu) / (variance + eps)) - 1
    )
    return grad_mu, grad_sigma


def grad_entropy(
    theta:np.ndarray,
    state:np.ndarray,
    action:np.ndarray,
    grad_theta_mu:np.ndarray,
    grad_theta_sigma:np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    log_pi = np.log(pi(
        theta=theta,
        state=state,
        action=action
    ))

    grad_entropy_mu = -log_pi * grad_theta_mu
    grad_entropy_sigma = -log_pi * grad_theta_sigma

    return grad_entropy_mu, grad_entropy_sigma


def evaluation_play(theta:np.ndarray, coder:TileCoder):
    env = gym.make("InvertedPendulum-v5", render_mode='human')
    env.metadata["render_fps"] = 1
    terminated = False
    state = env.reset()[0]
    state = get_features(state, coder)
    total_reward = 0
    while not terminated:
        mu, sigma = get_mu_sigma(theta, state)
        action = np.random.normal(mu, sigma)
        state, reward, terminated, _, _ = env.step(
            np.clip(action, -3, 3)
        )
        state = get_features(state, coder)
        total_reward += reward

    env.close()
    return total_reward


def Actor_critic(
    env:gym.Env,
    theta:np.ndarray,
    w:np.ndarray,
    coder:TileCoder,
    steps:int,
    alpha_theta:float,
    alpha_w:float,
    entropy_coeff:float=0.0
):
    terminated, truncated = True, True
    pbar = tqdm(range(steps))
    rewards_per_episode = []
    episode_rewards = 0
    for step in pbar:
        if terminated or truncated:
            state, _ = env.reset()
            state = get_features(state, coder=coder)
            terminated, truncated = False, False
            rewards_per_episode.append(episode_rewards)
            episode_rewards = 0

        mu, sigma = get_mu_sigma(theta, state)
        action = np.random.normal(mu, sigma)
        new_state, reward, terminated, truncated, _ = env.step(
            np.clip(action, -3, 3)
        )
        new_state = get_features(new_state, coder=coder)
        episode_rewards += reward

        v_state = state.T@w
        delta = reward - v_state.squeeze()

        if not terminated:
            delta += new_state.T@w

        w = w + (alpha_w * delta * state)[..., None]

        theta_mu , theta_sigma = np.split(theta, 2)
        grad_mu, grad_sigma = grad_ln_pi(
            theta=theta,
            state=state,
            action=action
        )

        theta_mu = theta_mu + alpha_theta * delta * grad_mu
        theta_sigma = theta_sigma + alpha_theta * delta * grad_sigma

        entropy_mu, entropy_sigma = grad_entropy(
            theta=theta,
            state=state,
            action=action,
            grad_theta_mu=grad_mu,
            grad_theta_sigma=grad_sigma
        )

        #theta_mu = theta_mu + alpha_theta * entropy_coeff * entropy_mu
        theta_sigma = theta_sigma + alpha_theta * entropy_coeff * entropy_sigma

        theta = np.vstack([theta_mu, theta_sigma])
        state = new_state

        if step % 1000 == 0:
            pbar.set_postfix({'episode_reward':rewards_per_episode[-1]})

    return theta, np.array(rewards_per_episode, dtype=np.float32)


def Differential_Actor_critic(
    env:gym.Env,
    theta:np.ndarray,
    w:np.ndarray,
    coder:TileCoder,
    steps:int,
    alpha_theta:float,
    alpha_w:float,
    alpha_r:float,
    entropy_coeff:float=0.0
):
    terminated, truncated = True, True
    r = 0
    pbar = tqdm(range(steps))
    rewards_per_episode = []
    episode_rewards = 0
    for step in pbar:
        if terminated or truncated:
            state, _ = env.reset()
            state = get_features(state, coder=coder)
            terminated, truncated = False, False
            rewards_per_episode.append(episode_rewards)
            episode_rewards = 0

        mu, sigma = get_mu_sigma(theta, state)
        action = np.random.normal(mu, sigma)
        new_state, reward, terminated, truncated, _ = env.step(
            np.clip(action, -3, 3)
        )
        new_state = get_features(new_state, coder=coder)
        episode_rewards += reward

        v_state = state.T@w
        delta = reward - r - v_state.squeeze()

        if not terminated:
            delta += new_state.T@w

        r = r + alpha_r * delta
        w = w + (alpha_w * delta * state)[..., None]

        theta_mu , theta_sigma = np.split(theta, 2)
        grad_mu, grad_sigma = grad_ln_pi(
            theta=theta,
            state=state,
            action=action
        )

        theta_mu = theta_mu + alpha_theta * delta * grad_mu
        theta_sigma = theta_sigma + alpha_theta * delta * grad_sigma

        entropy_mu, entropy_sigma = grad_entropy(
            theta=theta,
            state=state,
            action=action,
            grad_theta_mu=grad_mu,
            grad_theta_sigma=grad_sigma
        )

        #theta_mu = theta_mu + alpha_theta * entropy_coeff * entropy_mu
        theta_sigma = theta_sigma + alpha_theta * entropy_coeff * entropy_sigma

        theta = np.vstack([theta_mu, theta_sigma])
        state = new_state

        if step % 1000 == 0:
            pbar.set_postfix({'episode_reward':rewards_per_episode[-1]})

    return theta, np.array(rewards_per_episode, dtype=np.float32)

if __name__ == '__main__':
    env = gym.make("InvertedPendulum-v5")

    coder = TileCoder(
        low=[-5.0, -1.0, -5.0, -1.0],
        high=[5.0, 1.0, 5.0, 1.0],
        num_tiles=8,
        tile_per_dimension=[8, 8, 8, 8]
    )

    theta = np.zeros((coder.num_features * 2, 1), dtype=np.float32)
    w = np.zeros((coder.num_features, 1), dtype=np.float32)

    theta, actor_critic_rpe = Actor_critic(
        env=env,
        theta=theta,
        w=w,
        coder=coder,
        steps=500_000,
        alpha_theta=1e-4,
        alpha_w=0.2/8,
        entropy_coeff=0.001
    )

    total_rewards = evaluation_play(theta, coder)
    print('Evaluation reward:', total_rewards)

    theta = np.zeros((coder.num_features * 2, 1), dtype=np.float32)
    w = np.zeros((coder.num_features, 1), dtype=np.float32)

    theta, diff_actor_critic_rpe = Differential_Actor_critic(
        env=env,
        theta=theta,
        w=w,
        coder=coder,
        steps=500_000,
        alpha_theta=1e-4,
        alpha_r=0.2/8,
        alpha_w=0.2/8,
        entropy_coeff=0.001
    )

    total_rewards = evaluation_play(theta, coder)
    print('Evaluation reward:', total_rewards)


    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(diff_actor_critic_rpe, label='diff_actor_critic')
    ax.plot(actor_critic_rpe, label='actor_critic')
    ax.grid(c="#eee")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title("Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.savefig("actor_critic_entropy.jpeg", dpi=310)