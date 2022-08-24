import copy
import time
from collections import deque, namedtuple
from random import random, choices

import matplotlib.pyplot as plt

import gym
import torch
from gym import wrappers
from numpy.random import randint

from torch import nn, optim
import numpy as np


class ReplayBuffer:
    Sample = namedtuple('Sample', 'prev_state action reward state done')

    def __init__(self, maximum_length: int):
        self.maximum_length = maximum_length
        self.samples = deque(maxlen=maximum_length)

    def __len__(self):
        return len(self.samples)

    def __bool__(self):
        return len(self.samples) == self.maximum_length

    def append(self, prev_state, action, reward, state, done):
        sample = self.Sample(prev_state, action, reward, state, done)
        self.samples.append(sample)

    def sample(self, samples_size: int):
        batch = choices(self.samples, k=samples_size)
        states = torch.cat([b[0] for b in batch])
        actions = torch.Tensor([b[1] for b in batch])
        rewards = torch.Tensor([b[2] for b in batch])
        next_states = torch.cat([b[3] for b in batch])
        done = torch.Tensor([b[4] for b in batch])

        return states, actions, rewards, next_states, done


class DQNAgent:

    def __init__(self, observation_space: int, action_space: int, epsilon: float, epsilon_reduce_factor: float,
                 num_of_neurons: int = 256, num_of_layers: int = 2, lr: float = 5e-5, gamma: float = 0.99,
                 scheduler_t_max=None):
        self.action_space = action_space
        self.observation_space = observation_space
        layers = [
            nn.Linear(observation_space, num_of_neurons),
            nn.ReLU(),
        ]
        for i in range(1, num_of_layers):
            layers.append(nn.Linear(num_of_neurons, num_of_neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(num_of_neurons, action_space))
        self.policy_model = nn.Sequential(*layers)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.target_model = copy.deepcopy(self.policy_model)
        self.update_target_model()

        self.policy_model.to(device)

        for p in self.policy_model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_reduce_factor = epsilon_reduce_factor
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=scheduler_t_max, eta_min=0)

    def get_action(self, state, explore: bool = True):
        if not explore or random() > self.epsilon:
            q_action = self.policy_model(state)
            action = torch.argmax(q_action).item()
        else:
            action = randint(0, self.action_space)
        return action

    def reduce_exploration(self):
        self.epsilon *= self.epsilon_reduce_factor
        print('Setting epsilon to:', self.epsilon)

    def train_on_batch(self, states, actions, rewards, next_states, done):

        with torch.no_grad():
            q_next = self.target_model(next_states)
            actual = rewards + self.gamma * ((1 - done) * torch.max(q_next, dim=1)[0])

        q = self.policy_model(states)
        predicted = q.gather(dim=1, index=actions.long().unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(predicted, actual)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())


class Trainer:

    def __init__(self, environment, epochs: int, reduce_epsilon_epoch: int, reduce_epsilon_steps: int,
                 replay_buffer: int = 2000, start_epsilon: float = 1.0, end_epsilon: float = 0.1,
                 look_back_num: int = 1, **kwargs):
        self.env = environment
        self.epochs = epochs
        self.look_back_num = look_back_num
        self.observation_space = self.env.observation_space.shape[0] * look_back_num
        self.action_space = self.env.action_space.n
        self.replay_buffer = ReplayBuffer(replay_buffer)
        self.counter = 0
        self.reduce_epsilon_epoch = reduce_epsilon_epoch
        epsilon_reduce_factor = (end_epsilon / start_epsilon) ** (1 / reduce_epsilon_steps)
        self.epochs_to_reduce = list(range(0, reduce_epsilon_epoch + 1, reduce_epsilon_epoch // reduce_epsilon_steps))[
                                1:]
        self.agent = DQNAgent(self.observation_space, self.action_space, start_epsilon, epsilon_reduce_factor,
                              scheduler_t_max=epochs // 3, **kwargs)

    def _train_single_episode(self, batch_size: int):
        state_observation = self.env.reset()
        episode_reward = 0
        state = np.concatenate([state_observation] * self.look_back_num)
        while True:

            state_ = torch.from_numpy(state).reshape(1, -1)
            action = self.agent.get_action(state_)

            prev_state = state_
            state_observation, reward, episode_done, _ = self.env.step(action)
            episode_reward += reward
            state = np.concatenate([state, state_observation])[len(state_observation):]
            _state = torch.from_numpy(state).reshape(1, -1)

            self.replay_buffer.append(prev_state, action, reward, _state, episode_done)

            if self.replay_buffer:
                states, actions, rewards, next_states, done = self.replay_buffer.sample(batch_size)
                states += torch.normal(0, 1e-5, size=states.shape)
                self.agent.train_on_batch(states, actions, rewards, next_states, done)
                if self.counter % 50 == 0:
                    self.agent.update_target_model()

                self.counter += 1

            if episode_done:
                break

        print('reward', episode_reward)
        return episode_reward

    def train(self, batch_size: int):
        rewards = []
        for i in range(self.epochs):
            print('Epoch', i)
            total_episode_reward = self._train_single_episode(batch_size)

            if i in self.epochs_to_reduce:
                self.agent.reduce_exploration()

            rewards.append(total_episode_reward)

        return rewards


if __name__ == '__main__':

    ENV_NAME = 'CartPole-v1'
    EPOCHS = 1000
    REDUCE_EPSILON_EPOCH = EPOCHS * 3 // 4
    REDUCE_EPSILON_STEPS = 4
    END_EPSILON = 0.01
    BATCH_SIZE = 256
    GAMMA = 0.99
    REPLAY_BUFFER = 1000
    LOOK_BACK_NUM = 1

    video_path = f'c:/videos/{ENV_NAME}_{LOOK_BACK_NUM}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'

    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, video_path, resume=True, video_callable=lambda count: count % 50 == 0)

    trainer = Trainer(env, EPOCHS, REDUCE_EPSILON_EPOCH, REDUCE_EPSILON_STEPS, end_epsilon=END_EPSILON,
                      replay_buffer=REPLAY_BUFFER, look_back_num=LOOK_BACK_NUM,
                      num_of_layers=2, num_of_neurons=256, lr=5e-5, gamma=GAMMA)
    rewards = trainer.train(BATCH_SIZE)

    plt.plot(rewards)
    plt.show()
