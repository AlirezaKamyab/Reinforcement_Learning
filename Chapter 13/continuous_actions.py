import numpy as np
import gymnasium as gym
from tqdm import tqdm
from tile_coding import TileCoder


def softmax(value:np.ndarray) -> np.ndarray:
    logits = np.exp(value)
    return logits / np.sum(logits)


def get_mu_sigma(state:np.ndarray, theta:np.ndarray):
    theta_mu, theta_sigma = np.split(theta, 2)
    mu = theta_mu.T@state
    sigma = np.exp(theta_sigma.T @ state)

    return mu, sigma


def pi(state:np.ndarray, a:np.ndarray, theta:np.ndarray):
    mu, sigma = get_mu_sigma(state=state, theta=theta)
    epsilon = 1e-9
    return (1 / (sigma * np.sqrt(2 * np.pi) + epsilon)) * np.exp(-np.square(a - mu) / (2 * np.square(sigma) + epsilon))


def grad_ln_pi(state:np.ndarray, action:np.ndarray, theta:np.ndarray):
    mu, sigma = get_mu_sigma(state=state, theta=theta)

    epsilon = 1e-9
    grad_mu = np.einsum('a,d->da', (1 / (np.square(sigma) + epsilon)) * (action - mu), state)
    grad_sigma = np.einsum('a,d->da', (np.square(action - mu) / (np.square(sigma) + epsilon)) - 1, state)

    return grad_mu, grad_sigma


def grad_entropy(state:np.ndarray, action:np.ndarray, theta:np.ndarray):
    grad_mu, grad_sigma = grad_ln_pi(state=state, action=action, theta=theta)
    grad_entropy_mu = -np.log(pi(state=state, a=action, theta=theta)) * grad_mu
    grad_entropy_sigma = -np.log(pi(state=state, a=action, theta=theta)) * grad_sigma
    return grad_entropy_mu, grad_entropy_sigma


def actor_critic_continuous(
    env:gym.Env,
    theta:np.ndarray,
    w:np.ndarray,
    coder:TileCoder,
    alpha_w:float,
    alpha_theta:float,
    alpha_r:float,
    gamma:float=1.0,
    steps:int=1,
    entropy_coeff:float=0.0
):
    r_pi = 0
    terminated = True
    truncated = False
    num_completes = 0

    for _ in range(steps):
        if terminated or truncated:
            state, _ = env.reset()
            state = coder.get_vector(state)
            terminated = False
            truncated = False
        mu, sigma = get_mu_sigma(state, theta)
        raw_action = np.random.normal(loc=mu, scale=sigma)
        action = np.clip(raw_action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = coder.get_vector(next_state)

        if not terminated:
            target = reward - r_pi + gamma * w.T@next_state
        else:
            target = reward - r_pi
            num_completes += 1

        delta = target - w.T@state
        r_pi = r_pi + alpha_r * delta
        w = w + alpha_w * delta * state

        # grad theta
        theta_mu, theta_sigma = np.split(theta, 2)
        grad_mu, grad_sigma = grad_ln_pi(state, raw_action, theta)
        grad_entropy_mu, grad_entropy_sigma = grad_entropy(state=state, action=raw_action, theta=theta)
        theta_mu = theta_mu + alpha_theta * (
            gamma * delta * grad_mu + entropy_coeff * grad_entropy_mu
        )

        theta_sigma = theta_sigma + alpha_theta * (
            gamma * delta * grad_sigma + entropy_coeff * grad_entropy_sigma
        )

        theta = np.vstack((theta_mu, theta_sigma))

        state = next_state

    return theta, w, num_completes

if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")

    coder = TileCoder(
        low=[-1.2, -0.07],
        high=[0.6, 0.07],
        num_tiles=8,
        tile_per_dimension=[8, 8]
    )
    theta = np.zeros((2 * coder.num_features, 1), dtype=np.float64)
    w = np.zeros((coder.num_features,), dtype=np.float32)

    runs = 10
    for ec in [0.0, 1.0, 0.1, 0.001, 0.0001]:
        mean_completes = 0
        for _ in tqdm(range(runs)):
            _, _, num_completes = actor_critic_continuous(
                env=env,
                theta=theta,
                w=w,
                coder=coder,
                alpha_r=0.1,
                alpha_theta=1e-4,
                alpha_w=0.1,
                entropy_coeff=ec,
                steps=200000
            )
            mean_completes += num_completes / runs
        print(f'Entropy Coeff is {ec} and mean of the number of completes is {mean_completes}')
