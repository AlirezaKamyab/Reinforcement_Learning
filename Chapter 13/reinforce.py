#!/usr/bin/env python3
import numpy as np
from typing import List
from short_corridor import ShortCorridorEnv


def x(state:int, action:int) -> np.ndarray:
    features = np.array([[1, 0], [0, 1]])
    return features[action]


def state_x(state:int, num_states:int=4) -> np.ndarray:
    vec = np.zeros(num_states)
    vec[state] = 1.0
    return vec


def h(state:int, action:int, theta:np.ndarray) -> float:
    return theta.T@x(state, action)


def pi(action:int, state:int, theta:np.ndarray) -> float:
    logits = np.array([h(state, a, theta) for a in (0, 1)], dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs[action] / probs.sum()


def grad_ln_pi(action:int, state:int, theta:np.ndarray) -> np.ndarray:
    S = np.sum([pi(b, state, theta) * x(state, b) for b in (0, 1)])
    return x(state, action) - S


def generate_trajectory(env:ShortCorridorEnv, theta:np.ndarray) -> List[dict]:
    state = env.reset()
    trajectory = [{'state':state}]

    while not env.terminated:
        actions_p = [pi(a, state, theta) for a in (0, 1)]
        action = int(np.random.choice([0, 1], p=actions_p))
        reward, state, _ = env.step(action)

        trajectory[-1]['action'] = action
        trajectory.append({'state':state, 'reward':reward})
    return trajectory


def reinforce(
    env:ShortCorridorEnv,
    theta:np.ndarray,
    episodes:int,
    alpha:float,
    gamma:float=1.0
):
    rewards_per_episode = []
    for _ in range(episodes):
        trajectory = generate_trajectory(env=env, theta=theta)
        rewards_per_episode.append(sum(x['reward'] for x in trajectory[1:]))

        for t in range(len(trajectory) - 1):
            G = sum(gamma ** i * x['reward'] for i, x in enumerate(trajectory[t + 1:]))

            state = trajectory[t]['state']
            action = trajectory[t]['action']
            theta = theta + alpha * gamma ** t * G * grad_ln_pi(action, state, theta)

    return {
        'theta':theta,
        'rewards_per_episode': rewards_per_episode
    }


def reinforce_with_baseline(
    env:ShortCorridorEnv,
    theta:np.ndarray,
    w:np.ndarray,
    episodes:int,
    alpha_theta:float,
    alpha_w:float,
    gamma:float=1.0
):
    rewards_per_episode = []

    for _ in range(episodes):
        trajectory = generate_trajectory(env=env, theta=theta)
        rewards_per_episode.append(sum([x['reward'] for x in trajectory[1:]]))

        for t in range(len(trajectory) - 1):
            G = np.sum([gamma ** i * x['reward'] for i, x in enumerate(trajectory[t+1:])])

            state = trajectory[t]['state']
            action = trajectory[t]['action']
            delta = G - w.T@state_x(state)
            grad_w = state_x(state)

            w = w + alpha_w * delta * grad_w
            theta = theta + alpha_theta * (gamma ** t) * delta * grad_ln_pi(action, state, theta)

    return {
            'theta':theta,
            'rewards_per_episode': rewards_per_episode
        }



if __name__ == '__main__':
    theta = np.array([-1.47, 1.47])
    env = ShortCorridorEnv()
    w = np.zeros(env.num_states)
    output = reinforce_with_baseline(
        env=env,
        theta=theta,
        w=w,
        episodes=20000,
        alpha_theta=2**(-13),
        alpha_w=0.01
    )

    print(output)
    theta = output['theta']
    actions = [pi(a, 0, theta) for a in (0, 1)]
    print("actions:", actions)