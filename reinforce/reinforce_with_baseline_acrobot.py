import os.path
from datetime import datetime
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from torch.optim.lr_scheduler import ExponentialLR


class ReinforceNet(nn.Module):
    def __init__(self,
                 structure: Tuple[Tuple[int]],
                 policy_head_structure: Tuple[Tuple[int]],
                 value_head_structure: Tuple[Tuple[int]],
                 device):
        super(ReinforceNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(item[0], item[1]) for item in structure])
        self.policy_head = nn.ModuleList([nn.Linear(item[0], item[1]) for item in policy_head_structure])
        self.value_head = nn.ModuleList([nn.Linear(item[0], item[1]) for item in value_head_structure])

        self.to(device)

        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        policy = x
        for i, layer in enumerate(self.policy_head):
            policy = layer(policy)
            if i < len(self.policy_head) - 1:
                policy = F.relu(policy)
        policy = F.log_softmax(policy, dim=-1)

        value = x
        for i, layer in enumerate(self.value_head):
            value = layer(value)
            if i < len(self.value_head) - 1:
                value = F.relu(value)

        return policy, value


class REINFORCEAgent:
    def __init__(self,
                 environment,
                 hidden_structure: Tuple[Tuple[int]],
                 policy_head_structure: Tuple[Tuple[int]],
                 value_head_structure: Tuple[Tuple[int]],
                 lr: float,
                 gamma: float,
                 device,
                 look_back: int = 1):
        self.environment = environment
        self.look_back = look_back
        self.gamma = gamma
        self.network = ReinforceNet(hidden_structure, policy_head_structure, value_head_structure, device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_fun = torch.nn.MSELoss()
        self.device = device

    def learn(self, epoch, video):
        state, _ = self.environment.reset()
        state = np.concatenate([state] * self.look_back)

        rewards_sum = 0
        rewards = []
        probabilities = []
        state_values = []
        counter = 0
        while True:
            if video:
                video.capture_frame()
            state = torch.from_numpy(state).flatten().to(self.device)
            actions_dist, state_value = self.network(state)
            state_values.append(state_value)
            action_dist = torch.distributions.Categorical(logits=actions_dist)
            action = action_dist.sample()
            probability = actions_dist[action]
            probabilities.append(probability)

            state, reward, episode_done, truncated, _ = env.step(action.item())
            rewards.append(reward)
            rewards_sum += reward
            counter += 1
            if episode_done or truncated:
                break

        probabilities = torch.stack(probabilities).flip(dims=(0,)).to(self.device)
        state_values = torch.stack(state_values).flip(dims=(0,)).flatten().to(self.device)

        returns = torch.Tensor(rewards).flip(dims=(0,)).to(self.device)
        for i in range(1, len(returns)):
            returns[i] = returns[i] + self.gamma * returns[i - 1]

        advantage = returns - state_values.detach()
        policy_loss = (- probabilities * advantage).sum()

        value_loss = self.loss_fun(state_values, returns)

        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.optimizer.step()

        if epoch + 1 % 100 == 0:
            self.scheduler.step()

        return rewards_sum


if __name__ == '__main__':
    epochs = 1001
    look_back = 1
    lr = 0.00099
    gamma = 0.99

    torch.manual_seed(123)
    np.random.seed(123)

    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    video_path = os.path.join('.', 'videos', os.path.basename(__file__).split('.')[0],
                              datetime.now().strftime("%Y%m%d_%H%M%S"))
    Path(video_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)

    observation_space = env.observation_space.shape[0] * look_back
    action_space = env.action_space.n
    network_topology = {
        'hidden_structure': ((observation_space, 115), (115, 115)),
        'policy_head_structure': ((115, action_space),),
        'value_head_structure': ((115, 115), (115, 115), (115, 115), (115, 1)),
    }
    agent = REINFORCEAgent(env, lr=lr, look_back=look_back, gamma=gamma, device=device, **network_topology)

    epoch_rewards = []
    for epoch in range(epochs):
        video = None
        if epoch % 100 == 0:
            video = VideoRecorder(env, os.path.join(video_path, f'epoch_{epoch}.mp4'))
            print('epoch:', epoch)

        try:
            total_reward = agent.learn(epoch, video)
            epoch_rewards.append(total_reward)
        finally:
            if video:
                video.close()
        print('total reward', total_reward, 'mean reward', np.mean(epoch_rewards))

    env.close()
    plt.plot(epoch_rewards)
    plt.show()