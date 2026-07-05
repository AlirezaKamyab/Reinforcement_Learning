import collections
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm
import cv2
from torch.utils.tensorboard.writer import SummaryWriter


gym.register_envs(ale_py)
env = gym.make('ALE/Pong-v5', render_mode='rgb_array', frameskip=1)
env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
env = gym.wrappers.AtariPreprocessing(
    env, 
    frame_skip=1,
    screen_size=(84, 84), 
    scale_obs=False, 
    grayscale_obs=True,
    terminal_on_life_loss=True
)
env = gym.wrappers.FrameStackObservation(env, 4) # Stacks 4 frames


class Q_network(nn.Module):
    def __init__(self, input_features:int=3, num_actions:int=4):
        super(Q_network, self).__init__()
        self.input_features = input_features
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(
            in_channels=input_features, 
            out_channels=32, 
            kernel_size=(8, 8),
            stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(4, 4),
            stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=(3, 3),
            stride=1
        )

        self.linear = nn.Linear(7 * 7 * 64, 512)
        self.action_mapper = nn.Linear(512, num_actions)

    def forward(self, x:torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear(x))
        x = self.action_mapper(x)
        return x
    

class ReplayBuffer:
    def __init__(self, max_capacity:int=10000, device:str='cpu'):
        self.buffer = collections.deque(maxlen=max_capacity)
        self.device = device
    
    def insert(
        self,
        state:torch.Tensor,
        action:torch.Tensor,
        reward:torch.Tensor,
        next_state:torch.Tensor,
        done:torch.Tensor
    ):
        state = state * 255.0
        next_state = next_state * 255.0
        self.buffer.append((
            state.detach().to(dtype=torch.uint8, device='cpu'), 
            action.detach().to(dtype=torch.uint8, device='cpu'), 
            reward.detach().to(dtype=torch.int8, device='cpu'), 
            next_state.detach().to(dtype=torch.uint8, device='cpu'), 
            done.detach().to(dtype=torch.uint8, device='cpu')
        ))

    def sample(self, batch_size:int):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.concat(states, dim=0).to(dtype=torch.float32, device=self.device)
        actions = torch.concat(actions, dim=0).to(dtype=torch.int64, device=self.device)
        rewards = torch.concat(rewards, dim=0).to(dtype=torch.float32, device=self.device)
        next_states = torch.concat(next_states, dim=0).to(dtype=torch.float32, device=self.device)
        dones = torch.concat(dones, dim=0).to(dtype=torch.float32, device=self.device)

        # scale the states
        states = states / 255.0
        next_states = next_states / 255.0

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    

class DQN:
    def __init__(
        self,
        env:gym.Env,
        q_network:Q_network,
        num_actions:int=4,
        init_alpha:float=0.01,
        epsilon:float=1.0,
        epsilon_decay_until:float=500_000,
        min_epsilon:float=0.1,
        gamma:float=0.99,
        replay_capacity:int=10000,
        min_replay:int=128,
        batch_size:int=32,
        input_features:int=3,
        steps_to_swap_target:int=256,
        device:str='cpu'
    ):
        self.env = env
        self.num_actions = num_actions
        self.q_network = q_network
        # epsilon
        self.epsilon = epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.min_epsilon = min_epsilon

        # Replay buffer
        self.replay_capacity = replay_capacity
        self.min_replay = min_replay
        self.replay_buffer = ReplayBuffer(replay_capacity, device=device)

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_to_swap_target = steps_to_swap_target
        self.input_features = input_features
        self.init_alpha = init_alpha
        self.alpha = init_alpha

        # Logging parameters
        self.last_loss = 0.0
        self.steps_taken = 0

        # Models
        self.device=device
        self.target_network = Q_network(
            input_features=q_network.input_features,
            num_actions=q_network.num_actions
        ).to(device=device)
        self.target_network.load_state_dict(q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.RMSprop(
            self.q_network.parameters(), 
            lr=self.alpha,
            momentum=0.95, 
            alpha=0.95,
            eps=0.01
        )
        self.summary_writer = SummaryWriter()

    def decay_epsilon(self):
        self.epsilon = (self.min_epsilon - 1) / self.epsilon_decay_until * self.steps_taken + 1
        self.epsilon = max(self.min_epsilon, self.epsilon)
        self.summary_writer.add_scalar('epsilon', self.epsilon, global_step=self.steps_taken)

    def convert_state_to_torch(self, state:np.ndarray):
        state = torch.tensor(state, dtype=torch.float32)
        state = state.unsqueeze(0)
        state = state.to(dtype=torch.float32, device=self.device)
        state = state / 255.0
        return state

    def choose_action(self, state:torch.Tensor, epsilon:float=0.0):
        # state has the shape [B, 3, W, H]
        if np.random.rand() < epsilon:
            actions = np.random.choice(self.num_actions, size=(state.shape[0], 1))
            return torch.tensor(actions, dtype=torch.float32)
        
        with torch.no_grad():
            # q_values has the shape [B, 4]
            self.q_network.eval()
            q_values = self.q_network(state)
            max_q = torch.max(q_values, dim=1, keepdim=True)
            return max_q.indices
        
    def backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
    def step(
            self, 
            state:torch.Tensor,
            actions:torch.Tensor,
            rewards:torch.Tensor,
            next_states:torch.Tensor,
            done:torch.Tensor
        ):
        self.q_network.train()
        with torch.no_grad():
            q_values = self.target_network(next_states)
            max_q = torch.max(q_values, dim=1, keepdim=True).values
            td_target = (rewards + self.gamma * max_q * (1 - done))

        current_values = self.q_network(state)
        current_values = torch.gather(current_values, dim=1, index=actions)

        loss = F.mse_loss(current_values, td_target)

        self.summary_writer.add_scalar(
            "TD_error",
            scalar_value=(td_target - current_values).mean(),
            global_step=self.steps_taken
        )

        self.last_loss = loss.detach().cpu().item()
        self.backpropagate(loss=loss)
        self.steps_taken += 1
        self.decay_epsilon()

        if (self.steps_taken + 1) % self.steps_to_swap_target == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()

    def run_episode(self):
        sum_rewards = 0
        state = self.convert_state_to_torch(
            self.env.reset()[0]
        )
        terminated = False
        while not terminated:
            t_action = self.choose_action(state, epsilon=self.epsilon).to(self.device)
            action = int(t_action.detach().cpu()[0][0])

            next_state, reward, terminated, _, _ = self.env.step(action)
            sum_rewards += reward

            next_state = self.convert_state_to_torch(next_state)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            reward = torch.clip(reward, min=-1, max=1)
            reward = reward.reshape((1, 1))
            done = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

            if terminated:
                done[0, 0] = 1.0

            self.replay_buffer.insert(
                    state=state,
                    action=t_action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
            
            if len(self.replay_buffer) < self.min_replay:
                state = next_state
                continue

            b_states, b_actions, b_rewards, b_next_states, b_done = self.replay_buffer.sample(self.batch_size)
            self.step(
                state=b_states,
                actions=b_actions,
                rewards=b_rewards,
                next_states=b_next_states,
                done=b_done
            )

            state = next_state
        return sum_rewards

    def run(self, episodes:int=1):
        with tqdm(range(episodes), leave=True) as pbar:
            for episode in pbar:
                rewards = self.run_episode()
                self.summary_writer.add_scalar("reward_per_episode", rewards, global_step=episode, new_style=True)
                self.summary_writer.add_scalar("buffer_size", len(self.replay_buffer), global_step=episode)
                pbar.set_postfix({"rewards":rewards, "epsilon":self.epsilon, 'steps':self.steps_taken, "loss":float(self.last_loss)})

                if episode % 1000 == 0:
                    self.generate_video(output_name=f"output_{episode}.mp4")
                    torch.save(self.q_network, 'q_network.pt')

    def generate_video(
        self,
        output_name:str='output.mp4',
        fps:float=40, 
        shape:tuple=(160, 210)
    ):
        # Play
        self.q_network.eval()
        frames = []
        state = self.env.reset()[0]
        frames.append(self.env.render())
        terminated = False

        while not terminated:
            state = self.convert_state_to_torch(state)
            with torch.no_grad():
                actions = self.q_network(state).cpu().numpy()[0]
            best_action = np.argmax(actions)
            state, _, terminated, _, _ = self.env.step(best_action)
            frames.append(self.env.render())

            if len(frames) > 5000: break

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_name, fourcc, fps, shape)
        for frame in frames:
            frame = frame[:, :, ::-1]
            video_writer.write(frame)
        video_writer.release()

        

dqn = DQN(
    env=env,
    q_network=Q_network(input_features=4).cuda(),
    device='cuda',
    input_features=4,
    init_alpha=0.00025,
    steps_to_swap_target=1000,
    epsilon_decay_until=50_000,
    min_epsilon=0.1,
    batch_size=32,
    replay_capacity=300_000,
    min_replay=50_000,
    gamma=0.99
)

dqn.run(100_000)