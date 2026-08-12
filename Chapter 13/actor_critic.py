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


def one_step_actor_critic(
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
        episode_reward = 0
        state = env.reset()

        I = 1
        while not env.terminated:
            actions_p = [pi(a, state, theta) for a in (0, 1)]
            action = np.random.choice((0, 1), p=actions_p)
            reward, next_state, terminated = env.step(action)
            episode_reward += reward

            if terminated:
                delta = reward - w.T@state_x(state)
            else:
                delta = reward + gamma * w.T@state_x(next_state) - w.T@state_x(state)

            w = w + alpha_w * delta * state_x(state)
            theta = theta + alpha_theta * I * delta * grad_ln_pi(action, state, theta)

            state = next_state
            I = gamma * I

        rewards_per_episode.append(episode_reward)

    return {
        'theta':theta,
        'w':w,
        'rewards_per_episode':rewards_per_episode
    }


def actor_critic_with_eligibility_traces(
    env:ShortCorridorEnv,
    theta:np.ndarray,
    w:np.ndarray,
    episodes:int,
    alpha_theta:float,
    alpha_w:float,
    lambda_theta:float,
    lambda_w:float,
    gamma:float=1.0
):
    rewards_per_episode = []

    for _ in range(episodes):
        episode_rewards = 0
        state = env.reset()

        z_theta = np.zeros_like(theta)
        z_w = np.zeros_like(w)
        I = 1

        while not env.terminated:
            actions_p = [pi(a, state, theta) for a in (0, 1)]
            action = np.random.choice((0, 1), p=actions_p)
            reward, next_state, terminated = env.step(action)
            episode_rewards += reward

            if terminated:
                delta = reward - w.T@state_x(state)
            else:
                delta = reward + gamma * w.T@state_x(next_state) - w.T@state_x(state)

            z_w = gamma * lambda_w * z_w + state_x(state)
            z_theta = gamma * lambda_theta * z_theta + I * grad_ln_pi(action, state, theta)

            w = w + alpha_w * delta * z_w
            theta = theta + alpha_theta * delta * z_theta

            I = gamma * I
            state = next_state

        rewards_per_episode.append(episode_rewards)
    return {
        'theta':theta,
        'w':w,
        'rewards_per_episode':rewards_per_episode
    }


if __name__ == '__main__':
    theta = np.array([-1.47, 1.47])
    env = ShortCorridorEnv()
    w = np.zeros(env.num_states)
    output = actor_critic_with_eligibility_traces(
        env=env,
        theta=theta,
        w=w,
        episodes=10000,
        alpha_theta=2**(-13),
        alpha_w=0.001,
        lambda_w=0.9,
        lambda_theta=0.9
    )

    print(output)
    theta = output['theta']
    actions = [pi(a, 0, theta) for a in (0, 1)]
    print("actions:", actions)