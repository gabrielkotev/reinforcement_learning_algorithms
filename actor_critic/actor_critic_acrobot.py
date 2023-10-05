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


class ActorCriticNet(nn.Module):
    def __init__(self,
                 structure: Tuple[Tuple[int]],
                 actor_head_structure: Tuple[Tuple[int]],
                 critic_head_structure: Tuple[Tuple[int]],
                 device):
        super(ActorCriticNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(item[0], item[1]) for item in structure])
        self.actor_head = nn.ModuleList([nn.Linear(item[0], item[1]) for item in actor_head_structure])
        self.critic_head = nn.ModuleList([nn.Linear(item[0], item[1]) for item in critic_head_structure])

        self.to(device)

        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        actor = x
        for i, layer in enumerate(self.actor_head):
            actor = layer(actor)
            if i < len(self.actor_head) - 1:
                actor = F.relu(actor)
        actor = F.log_softmax(actor, dim=-1)

        critic = x
        for i, layer in enumerate(self.critic_head):
            critic = layer(critic)
            if i < len(self.critic_head) - 1:
                critic = F.relu(critic)

        return actor, critic


class ACAgent:
    def __init__(self,
                 environment,
                 hidden_structure: Tuple[Tuple[int]],
                 actor_head_structure: Tuple[Tuple[int]],
                 critic_head_structure: Tuple[Tuple[int]],
                 lr: float,
                 gamma: float,
                 device,
                 look_back: int = 1):
        self.environment = environment
        self.look_back = look_back
        self.gamma = gamma
        self.network = ActorCriticNet(hidden_structure, actor_head_structure, critic_head_structure, device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.loss_fun = torch.nn.MSELoss()
        self.device = device

    def learn(self, epoch, video):
        state, _ = self.environment.reset()
        state = np.concatenate([state] * self.look_back)

        rewards_sum = 0
        counts = 0
        episode_done = False
        while not episode_done:
            if video:
                video.capture_frame()
            state_tensor = torch.from_numpy(state).flatten().to(self.device)
            log_probs, state_value = self.network(state_tensor)
            action_dist = torch.distributions.Categorical(logits=log_probs)
            action = action_dist.sample()
            log_prob = log_probs[action]

            _state, _, _, _, _ = self.environment.step(action.item())
            reward = abs(_state[4])
            if counts > 1000:
                episode_done = True
            state = np.concatenate([state, _state])[len(_state):]
            rewards_sum += reward

            next_state_tensor = torch.from_numpy(state).flatten().to(self.device)
            _, next_state_value = self.network(next_state_tensor)

            expected_state_value = reward + self.gamma * (1 - int(episode_done)) * next_state_value
            delta = expected_state_value - state_value

            actor_loss = -log_prob * delta
            critic_loss = self.loss_fun(expected_state_value, state_value)

            self.optimizer.zero_grad()
            total_loss = actor_loss + critic_loss
            total_loss.backward()
            self.optimizer.step()

            counts += 1

        if epoch + 1 % 100 == 0:
            self.scheduler.step()

        return reward


if __name__ == '__main__':
    epochs = 1501
    gamma = 0.90
    look_back = 3
    lr = 9.87898e-07

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
        'hidden_structure': ((observation_space, 299), (299, 299), (299, 299)),
        'actor_head_structure': ((299, 299), (299, 299), (299, 299), (299, action_space)),
        'critic_head_structure': ((299, 299), (299, 299), (299, 299), (299, 1)),
    }
    agent = ACAgent(env, lr=lr, gamma=gamma, look_back=look_back, device=device, **network_topology)

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
